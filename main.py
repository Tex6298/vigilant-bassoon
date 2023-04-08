from disarm_chatgpt import Disarm_GPT
from upload_disarm_data import upload_disarm_data

if __name__ == "__main__":
  upload_disarm_data()
  
  gpt = Disarm_GPT()
  qa_chain = gpt.get_chain()
  chat_history = []
  print("Chat with your docs!")
  while True:
    print("Human:")
    question = input()
    result = qa_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print("AI:")
    print(result["answer"])
