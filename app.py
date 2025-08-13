from flask import Flask, render_template, request
from llm_engine import get_response
from utils.search_device import search_device

app = Flask(__name__)

search_cache = []

@app.route("/", methods=["GET", "POST"])
@app.route("/device", methods=["GET", "POST"])
def device():
    r_m = None
    s_i = None
    queryu = ""
    if request.method == "POST":
        queryu = request.form.get("query")
        results = search_device(queryu)
        if results:
            search_cache.clear()
            r_m, s_i = get_response(queryu, results)
            search_cache.extend([queryu, r_m, s_i])
    if search_cache:
        queryu, r_m, s_i = search_cache  

    return render_template("device.html", result_markdown=r_m, speak_instructions=s_i, queryu=queryu)

@app.route("/about")
def about():
    return render_template("about.html")

application = app

# Local debug server (not used by Gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
