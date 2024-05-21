import { FireworksEmbeddings } from "@langchain/community/embeddings/fireworks";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Cohere } from "langchain/llms/cohere";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
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
    return;
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

  // insert data into pinecone index
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
    // create openai embeddings
    const embeddingsArrays = await getEmbeddings(chunks);

    console.log("Finished embedding documents");
    console.log(
      `Creating ${chunks.length} vectors array with id, values, and metadata...`
    );

    const batchSize = 100;
    let batch: any = [];
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
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
      if (batch.length === batchSize || i === chunks.length - 1) {
        console.log(`Inserting ${batch.length} vectors into pinecone index`);
        // upsert
        await index.upsert(batch);
        // empty batch
        batch = [];
      }
    }
  }
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
    console.log("Found matches, querying COHERE...", queryResponse.matches);
    const baseFireworkLLM = new Cohere({
      apiKey: process.env.COHERE_API_KEY || "",
    });
    const chain = await loadQAStuffChain(baseFireworkLLM, { verbose: true });
    const concatenatedPageContent = queryResponse.matches
      .map((match: any) => match.metadata.pageContent)
      .join(" ");
    const doc = new Document({ pageContent: concatenatedPageContent });

    const result = await chain.call({
      input_documents: [doc],
      question: question,
    });
    // save with fs
   
    return result.text
  } else {
    // 11. Log that there are no matches, so GPT-3 will not be queried
    console.log("Since there are no matches, COHERE will not be queried.");
  }

  return null;
};

async function getEmbeddings(chunks: any) {
  const fireworksEmbeddings = new FireworksEmbeddings({
    apiKey: process.env.OPENAI_API_KEY || "",
  });
  const documentRes = await fireworksEmbeddings.embedDocuments(
    chunks.map((chunk: any) => chunk.pageContent.replace(/\n/g, ""))
  );

  return documentRes;
}
