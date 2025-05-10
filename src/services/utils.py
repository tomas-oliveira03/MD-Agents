def formatUserInformation(userInformation: dict) -> str:
    formattedInfo = []
    
    for key, value in userInformation.items():
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        formattedInfo.append(f"{key}: {value}")
    
    # Join with comma only between entries, no leading comma
    result = formattedInfo[0] if formattedInfo else ""
    if len(formattedInfo) > 1:
        result += ", " + ", ".join(formattedInfo[1:])
    
    return result

        
def loadInitialPrompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    

def formatFinalPrompt(contextPrompt, userPrompt, context, userInformation):
    formattedUserInformation = formatUserInformation(userInformation)
    
    # Start building the prompt
    prompt = (
        f"{contextPrompt}\n\n"
        "Question:\n"
        f"{userPrompt}\n\n"
    )
    
    # Add user information if it exists, right after the question
    if formattedUserInformation:
        prompt += f"User Information:\n{formattedUserInformation}\n\n"
    
    # Add the articles context last
    prompt += f"Articles context:\n{context}"

    return prompt