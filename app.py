from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/hello', methods=['GET', 'POST'])
def hello():

    # POST request
    if request.method == 'POST':
        print('Incoming..')
        print(request.get_json())  # parse as JSON
        return 'OK', 200

    # GET request
    else:
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers

@app.route('/test')
def test_page():
    # look inside `templates` and serve `index.html`
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def homepage():
    message = 'it works'
    return message


if __name__ == "__main__":
    app.run()