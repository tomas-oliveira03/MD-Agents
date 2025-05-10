from LLMClient import LLMClient
from PineconeHandler import PineconeHandler
from utils import loadInitialPrompt, formatFinalPrompt

class Agent:
    
    def __init__(self, reasoningModel=True, contextPrompt="config/contextPrompt.txt", chunkedData="data/chunkedData.json", topK=5, targetThreshold=0.6, minimumThreshold=0.2, maxHierarchyLevel=3):
        self.contextPrompt = loadInitialPrompt(contextPrompt)
        self.pineconeHandler = PineconeHandler(chunkedData, topK, targetThreshold, minimumThreshold, maxHierarchyLevel)
        self.llmClient = LLMClient(reasoningModel)
        
    
    def handleRequest(self, requestId, prompt, userInformation):
        try:
            response = self.submitQuestion(prompt, userInformation)
            print(response)
        except Exception as error:
            print(f"Error handling request {requestId}: {error}")
            return None
        
        
    def submitQuestion(self, prompt, userInformation):
        # Retrieve relevant articles from Pinecone
        context = self.pineconeHandler.query(prompt)
        if context == "":
            raise Exception("The articles does not provide enough information to answer completely.")
        
        finalPrompt = formatFinalPrompt(self.contextPrompt, prompt, context, userInformation)   
        
        response = self.llmClient.generateResponse(finalPrompt)
        return response
            

if __name__ == "__main__":
    agent = Agent(reasoningModel=False)
    
    userPrompt = "How many hours of sleep should I get?"
    requestId = "123-242123-3213213"
    userInformation = {
        "age": 25,
        "weight": 70
    }

    agent.handleRequest(requestId, userPrompt, userInformation)

