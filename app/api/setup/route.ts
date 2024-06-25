import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { createPineconeIndex, updatePineconeIndex } from "../../../utils";
import { indexName } from "../../../config";
export async function POST() {
  const path = process.cwd()+"/documents";
  const loader = new DirectoryLoader(path, {
    ".txt": (path) => new TextLoader(path),
    ".md": (path) => new TextLoader(path),
    ".pdf": (path) => new PDFLoader(path),
  });

  const docs = await loader.load();
  const vectorDimensions = 768;

  const client = new Pinecone({ apiKey: process.env.PINECONE_API_KEY || "" });
  
  try {
    await createPineconeIndex(client, indexName, vectorDimensions);
    await updatePineconeIndex(client, indexName, docs);
  } catch (err) {
    console.log("error: ", err);
  }

  return NextResponse.json({
    data: "successfully created index and loaded data into pinecone...",
  });
}
