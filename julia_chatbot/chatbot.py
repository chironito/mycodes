import falcon
from juliacall import Main

Main.include("Chatbot.jl")

class PromptResource:
    def on_post(self, req, resp, path):
        session_id = path
        try:
            prompt = req.media['prompt']
        except KeyError:
            resp.status = falcon.HTTP_422
            resp.text = "No prompt in body"
            return
        resp.media = Main.Chatbot.llm_chat(prompt, session_id)
        resp.status = falcon.HTTP_200
        return

app = falcon.App()
app.add_route("/prompt/{path}", PromptResource())
