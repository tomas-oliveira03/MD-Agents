from services.LLMClient import LLMClient
from services.PineconeHandler import PineconeHandler
from services.utils import loadInitialPrompt, formatFinalPrompt, sendWebhook
from dotenv import load_dotenv
import os
import queue
import threading
import time


class Agent:
    
    def __init__(self, reasoningModel=True, contextPrompt="config/contextPrompt.txt", chunkedData="data/chunkedData.json", topK=5, targetThreshold=0.6, minimumThreshold=0.2, maxHierarchyLevel=3):
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
        
        ## TO REMOVE LATER
        userInformation = user["preferences"]
        print(userInformation)
        userHistory = {}
        
        finalPrompt = formatFinalPrompt(self.contextPrompt, prompt, context, userInformation)   
        print(f"\n\n{finalPrompt}\n\n")
        response = self.llmClient.generateResponse(finalPrompt, userHistory)
        return response
            

if __name__ == "__main__":
    agent = Agent(reasoningModel=False)
    
    requestId = "123-242123-3213213"
    user = {
        "name": "2",
        "age": 2
    },
    prompt = "How many hours of sleep should I get?"
    
    agent.handleRequest(requestId, user, prompt)
    
    while True:
        time.sleep(100)

    
