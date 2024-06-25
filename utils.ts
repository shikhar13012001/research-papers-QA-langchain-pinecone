import { FireworksEmbeddings } from "@langchain/community/embeddings/fireworks";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
// import fs from "fs";
import { ChatOpenAI } from "@langchain/openai";
import { indexName, timeout } from "./config";
import { type Pinecone, type IndexModel } from "@pinecone-database/pinecone";

// creating pinecone index

export function checkIfIndexExists(
  indexes: IndexModel[] = [],
  indexName: string
): boolean {
  for (const index of indexes) {
    if (index.name === indexName) {
      return true;
    }
  }

  return false;
}

export const createPineconeIndex = async (
  client: Pinecone,
  indexName: string,
  vectorDimension: number
) => {
  console.log("Checking Pinecone Index ", indexName);
  const existingIndexes = await client.listIndexes();
  if (checkIfIndexExists(existingIndexes.indexes, indexName)) {
    console.log("Index already exists");
    await client.deleteIndex(indexName);
  }
  console.log("Creating Index: ", indexName);
  const pineconeIndexPromise = client.createIndex({
    name: indexName,
    metric: "cosine",
    dimension: vectorDimension,
    spec: {
      serverless: {
        cloud: "aws",
        region: "us-east-1",
      },
    },
  });
  const timeoutPromise = new Promise((resolve, reject) => {
    setTimeout(() => {
      reject(new Error("Timeout while creating index"));
    }, timeout);
  });

  await Promise.race([pineconeIndexPromise, timeoutPromise]);
};

// update the pinecone index with the new data

export const updatePineconeIndex = async (
  client: Pinecone,
  indexName: string,
  docs: any
) => {
  // retrieve pinecone index
  const index = client.index(indexName);
  console.log("Updating Pinecone Index: ", indexName);
  // save to file docs.json and log the number of documents
  console.log(`Saving ${docs.length} documents to docs.json`);
  // save to file docs.json
  // fs.writeFileSync("docs.json", JSON.stringify(docs, null, 2));
  // insert data into pinecone index
  let chunksArr: any = [];
  for (const doc of docs) {
    console.log("Inserting document: ", doc.metadata.source);
    const txtPath = doc.metadata.source;
    const text = doc.pageContent;
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    console.log("Splitting text into chunks");
    const chunks = await textSplitter.createDocuments([text]);
    console.log("Embedding chunks: ", chunks.length);
    console.log(
      `Calling OpenAI's Embedding endpoint documents with ${chunks.length} text chunks ...`
    );
    const chunksObj = chunks.map((chunk: any) => {
      return {
        ...chunk,
        textPath: txtPath,
      };
    });
    // create openai embeddings
    chunksArr.push(chunksObj);
    // flatten the chunks
    chunksArr = chunksArr.flat();
  }
  console.log(
    "Finished chunking all documents",
    chunksArr.length,
    chunksArr[0]
  );

  // get embeddings array in 100 100 batches
  let embeddingsArrays: any = [];
  const promises: Promise<any>[] = [];
  for (let i = 0; i < chunksArr.length; i += 200) {
    const promise = getEmbeddings(chunksArr.slice(i, i + 200));
    promises.push(promise);
  }

  const results = await Promise.all(promises);
  embeddingsArrays = results.flat();
  // save to file embeddings.json
  // fs.writeFileSync(
  //   "embeddings.json",
  //   JSON.stringify(embeddingsArrays, null, 2)
  // );

  const batchSize = 250;

  let batch: any = [];
  let pineconePromises: Promise<any>[] = [];
  for (let i = 0; i < chunksArr.length; i++) {
    const chunk = chunksArr[i];
    const txtPath = chunk.textPath;
    delete chunk.textPath;
    const vector = {
      id: `${txtPath}-${i}`,
      values: embeddingsArrays[i],
      metadata: {
        ...chunk.metadata,
        loc: JSON.stringify(chunk.metadata.loc),
        pageContent: chunk.pageContent,
        txtPath: txtPath,
      },
    };
    batch.push(vector);
    if (batch.length === batchSize || i === chunksArr.length - 1) {
      console.log(`Inserting ${batch.length} vectors into pinecone index`);
      // upsert
      pineconePromises.push(index.upsert(batch));

      // empty batch
      batch = [];
    }
  }
  // wait for all promises to resolve
  await Promise.all(pineconePromises);
};

// query pinecone index

export const queryPineconeVectorStoreAndQueryLLM = async (
  client: Pinecone,
  indexName: string,
  question: string
) => {
  // 1. Start query process
  console.log("Querying Pinecone vector store...");
  // 2. Retrieve the Pinecone index
  const index = client.Index(indexName);
  // 3. Create query embedding
  const queryEmbedding = await new FireworksEmbeddings({
    apiKey: process.env.FIREWORKSAI_API_KEY || "",
  }).embedQuery(question);
  // 4. Query Pinecone index
  console.log("Querying Pinecone index...");
  let queryResponse = await index.query({
    topK: 10, // topK paragraphs or documents to return
    vector: queryEmbedding,
    includeMetadata: true,
    includeValues: true,
  });
  // 5. Log the number of matches
  console.log(`Found ${queryResponse.matches.length} matches...`);
  // 6. Log the question being asked
  console.log(`Asking question: ${question}...`);

  // if we found anything
  if (queryResponse.matches.length) {
    console.log(
      "Found matches, querying COHERE...",
      queryResponse.matches.length
    );
    // save the matches to a file
    // fs.writeFileSync(
    //   "matches.json",
    //   JSON.stringify(queryResponse.matches, null, 2)
    // );
    const model = new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0.2,
    });
    const prompt = ChatPromptTemplate.fromTemplate(
      `Use the following pieces of context to answer the question in plain text at the end in technical details, keep the answer crisp and to the point. If you don't know the answer, just say that you don't know, don't try to make up an answer: 
      {context}
      Question: {question}`
    );

    const chain = await createStuffDocumentsChain({
      llm: model,
      prompt,
    });
    const concatenatedPageContent = queryResponse.matches
      .map((match: any) => match.metadata.pageContent)
      .join(" ");
    const doc = new Document({ pageContent: concatenatedPageContent });
    console.log(doc);
    const result = await chain.invoke({
      context: [doc],
      question: question,
    });
    // save with fs

    return result;
  } else {
    // 11. Log that there are no matches, so GPT-3 will not be queried
    console.log("Since there are no matches, COHERE will not be queried.");
  }

  return null;
};

async function getEmbeddings(chunks: any) {
  const fireworksEmbeddings = new FireworksEmbeddings({
    apiKey: process.env.FIREWORKSAI_API_KEY || "",
  });
  const documentRes = await fireworksEmbeddings.embedDocuments(
    chunks.map((chunk: any) => chunk.pageContent.replace(/\n/g, ""))
  );

  return documentRes;
}
