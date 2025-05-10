from flask import jsonify, request
from services.Agent import Agent

specialistAgent = Agent(reasoningModel=False)

def registerSpecialistAgentRoutes(app, prefix):
    
    @app.route(f"{prefix}/ask", methods=["POST"])
    def askQuestion():
        data = request.get_json()
        requestId = data.get("requestId")
        userId = data.get("userId")
        userPrompt = data.get("userPrompt")

        if not requestId or not userId or not userPrompt:
            return jsonify({"error": "Missing requestId, userId, or userPrompt."}), 400
    
        # Get user information from the database
        userInformation = {
            "age": 25,
            "weight": 70
        }

        try:
            response = specialistAgent.submitQuestion(userPrompt, userInformation)
            return jsonify({
                "requestId": requestId,
                "response": response
            }), 200

        except Exception as e:
            return jsonify({
                "requestId": requestId,
                "error": str(e)
            }), 500
