from services.LLMClient import LLMClient
from services.PineconeHandler import PineconeHandler
from services.utils import loadInitialPrompt, formatPrompt, sendWebhook
from dotenv import load_dotenv
import os
import queue
import threading

class Agent:
    
    def __init__(self, reasoningModel=True, contextPrompt="src/config/contextPrompt.txt", chunkedData="src/data/chunkedData.json", topK=5, targetThreshold=0.6, minimumThreshold=0.2, maxHierarchyLevel=3):
        load_dotenv()
        globalOrchestratorBaseURL = os.getenv("GLOBAL_ORCHESTRATOR_BASE_URL")
        if not globalOrchestratorBaseURL:
            raise ValueError("GLOBAL_ORCHESTRATOR_BASE_URL environment variable not set.")
        self.globalOrchestratorEndpoint = globalOrchestratorBaseURL + "/reply"
        
        self.contextPrompt = loadInitialPrompt(contextPrompt)
        self.pineconeHandler = PineconeHandler(chunkedData, topK, targetThreshold, minimumThreshold, maxHierarchyLevel)
        self.llmClient = LLMClient(reasoningModel)
        
        # Create a queue and start a worker thread
        self.taskQueue = queue.Queue()
        self.workerThread = threading.Thread(target=self._processQueue, daemon=True)
        self.workerThread.start()
        
        
    def _processQueue(self):
        while True:
            try:
                requestId, user, prompt = self.taskQueue.get()
                print(f"[Worker] Processing request {requestId}")
                
                response = self.submitQuestion(prompt, user)
                print(f"[Worker] Response for {requestId}: {response}")
                
                # sendWebhook(self.globalOrchestratorEndpoint, {
                #     "requestId": requestId,
                #     "message": response
                # })
                
            except Exception as error:
                print(f"[Worker] Error handling request {requestId}: {error}")
                
                # sendWebhook(self.globalOrchestratorEndpoint, {
                #     "requestId": requestId,
                #     "error": str(error)
                # })
                
            finally:
                self.taskQueue.task_done()
                
    
    def handleRequest(self, requestId, user, prompt):
        self.taskQueue.put((requestId, user, prompt))
        print(f"Task added to queue for request {requestId}")
        
        
    def submitQuestion(self, prompt, user):
        # Retrieve relevant articles from Pinecone
        context = self.pineconeHandler.query(prompt)
        if context == "":
            raise Exception("The articles does not provide enough information to answer completely.")
        
        userInformation = user["preferences"]
        userHistory = user["conversation"]
        
        promptWithoutUserHistory = formatPrompt(self.contextPrompt, prompt, context, userInformation)   
        
        # If prompt is too long, automatically error out
        isValidRequest = self.llmClient.checkIfValidRequest(promptWithoutUserHistory)
        if not isValidRequest:
            raise Exception("The prompt received is too long.")
        
        # If not, we will check to see if we can add some user history as well
        promptWithUserHistory = self.checkMaxUserHistory(prompt, context, userInformation, userHistory)
        
        # If its not possible to add any history, just send the default prompt
        if promptWithUserHistory:
            finalPrompt = promptWithUserHistory
        else:
            finalPrompt = promptWithoutUserHistory
            
        print(f"\n\n{finalPrompt}\n\n")
        
        response = self.llmClient.generateResponse(finalPrompt)
        return response
            
    
    # Attempts to build a valid prompt using the full user history.
    # If the prompt is too long, it progressively removes the oldest entries
    # from userHistory (one at a time from the front) and retries.
    # Returns the first successfully validated prompt.
    # If no version of the prompt is valid with any history added, returns None.
    def checkMaxUserHistory(self, prompt, context, userInformation, userHistory):
        for i in range(len(userHistory) + 1):
            trimmedHistory = userHistory[i:]
            formattedPrompt = formatPrompt(self.contextPrompt, prompt, context, userInformation, trimmedHistory)

            if self.llmClient.checkIfValidRequest(formattedPrompt):
                return formattedPrompt

        return None
            
