import re

# Function to handle user input and generate responses
def chatbot_response(user_input):
    # Convert input to lowercase
    user_input = user_input.lower()

    # Pattern matching and response generation
    if re.search(r'\bhello\b|\bhi\b', user_input):
        return "Hello! How can I help you today?"
    
    elif re.search(r'\bhow are you\b', user_input):
        return "I'm just a chatbot, but I'm doing fine. How can I assist you?"
    
    elif re.search(r'\bwhat is your name\b', user_input):
        return "I am a chatbot created to assist you. I don't have a personal name."
    
    elif re.search(r'\bbye\b|\bgoodbye\b', user_input):
        return "Goodbye! Have a great day!"
    
    elif re.search(r'\b(what|how|why|where|when)\b', user_input):
        return "That's an interesting question. Let me find more information for you."
    
    elif re.search(r'\bthank you\b|\bthanks\b', user_input):
        return "You're welcome! If you have any more questions, feel free to ask."
    
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Main function to interact with the chatbot
def chat():
    print("Chatbot: Hi there! Type 'bye' to end the chat.")
    while True:
        user_input = input("You: ")
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
        
        if re.search(r'\bbye\b|\bgoodbye\b', user_input.lower()):
            break

# Start the chat
if __name__ == "__main__":
    chat()

        