package ht.mbds.salimbenzeragah.tp4web_salimbenzeragah.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiEmbeddingModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import jakarta.enterprise.context.Dependent;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Dependent
public class LlmClient implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private String systemRole;

    private transient Assistant assistant;
    private transient ChatMemory chatMemory;

    public LlmClient() {
        initialiserAssistant();
    }

    private void initialiserAssistant() {
        String apiKey = System.getProperty("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            apiKey = System.getenv("GEMINI_KEY");
        }
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("La variable GEMINI_KEY est absente.");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(
                RagResources.PDF_RAG,
                "Support de cours sur le RAG (Retrieval-Augmented Generation), " +
                        "les embeddings, les bases vectorielles et le fine-tuning."
        );

        sources.put(
                RagResources.PDF_IA,
                "Support de cours sur LangChain4j et l'Intelligence Artificielle : " +
                        "modèles de langage, AiServices, mémoire, outils, Guardrails et Jakarta EE."
        );

        // Fonctionnalité 1 (Test 2) : reformuler les questions contextuelles
        CompressingQueryTransformer queryTransformer = new CompressingQueryTransformer(model);

        // Fonctionnalité 2 (Test 3) : choisir le PDF pertinent avec Gemini
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, sources);

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .queryRouter(queryRouter)
                .build();

        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor)
                .build();
    }

    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;
        chatMemory.clear();
        if (systemRole != null && !systemRole.isBlank()) {
            chatMemory.add(SystemMessage.from(systemRole));
        }
    }

    public String chat(String question) {
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("La question ne peut pas être vide.");
        }
        return assistant.chat(question);
    }

    @Serial
    private void readObject(ObjectInputStream input) throws IOException, ClassNotFoundException {
        input.defaultReadObject();
        initialiserAssistant();
        if (systemRole != null && !systemRole.isBlank()) {
            chatMemory.add(SystemMessage.from(systemRole));
        }
    }

    private static final class RagResources {

        private static final EmbeddingModel EMBEDDING_MODEL = creerEmbeddingModel();

        private static final ContentRetriever PDF_RAG = creerRetriever("rag.pdf");

        private static final ContentRetriever PDF_IA = creerRetriever("ia.pdf");

        private static EmbeddingModel creerEmbeddingModel() {
            String apiKey = System.getProperty("GEMINI_KEY");
            if (apiKey == null || apiKey.isBlank()) {
                apiKey = System.getenv("GEMINI_KEY");
            }
            return GoogleGenAiEmbeddingModel.builder()
                    .apiKey(apiKey)
                    .modelName("gemini-embedding-001")
                    .build();
        }

        private static ContentRetriever creerRetriever(String fichier) {
            DocumentParser parser = new ApacheTikaDocumentParser();
            Document document = ClassPathDocumentLoader.loadDocument(fichier, parser);

            DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
            List<TextSegment> segments = splitter.split(document);

            List<Embedding> embeddings = EMBEDDING_MODEL.embedAll(segments).content();

            EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
            store.addAll(embeddings, segments);

            System.out.println("[RAG] Document indexé : " + fichier + " (" + segments.size() + " segments)");

            return EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(store)
                    .embeddingModel(EMBEDDING_MODEL)
                    .maxResults(5)
                    .minScore(0.4)
                    .build();
        }

        private RagResources() {
        }
    }
}