from flask import jsonify, request

from services.Agent import Agent

specialistAgent = Agent(reasoningModel=False)

def registerSpecialistAgentRoutes(app, prefix):
    
    @app.route(f"{prefix}/")
    def root():
        return jsonify({"message": "Welcome to the simplified API!"})

    @app.route(f"{prefix}/ask", methods=["POST"])
    def askQuestion():
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400

        # Use the global agent to process the question
        try:
            response = specialistAgent.submitQuestion(prompt)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
