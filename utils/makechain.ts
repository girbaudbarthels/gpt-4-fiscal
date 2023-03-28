import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Jij bent een AI assistant dat nuttig advies geeft over opzoekingen in het belgisch fiscaal recht. Je krijgt de volgende delen van een lang document en een vraag. Geef een conversationeel antwoord op basis van de gegeven context.
  Geef alleen hyperlinks naar de onderstaande context. Verzin GEEN hyperlinks.
  Als je het antwoord niet kunt vinden in de onderstaande context, zeg dan gewoon "Hmm, ik weet het niet zeker". Probeer geen antwoord te verzinnen.
  Als de vraag geen verband houdt met de context, antwoord dan beleefd dat je afgestemd bent om alleen vragen te beantwoorden die verband houden met de context.

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-4', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
          async handleLLMNewToken(token) {
            onTokenStream(token);
            console.log(token);
          },
        })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
