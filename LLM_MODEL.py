import google.generativeai as genai
import os
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

while (1):
    ques = str(input("\nfEnter a prompt here : "))
    print("--------------------GENERATIING RESPOSE--------------------")
    response = model.generate_content(ques)
    print("RESPONSE : \n",response.text, "\n")


