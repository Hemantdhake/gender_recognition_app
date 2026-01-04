from flask import Flask
from app import views

# app = Flask(__name__) # webserver gateway interphase (WSGI)
app = Flask(__name__, template_folder='template', static_folder='static')

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,
                 methods=['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask
# from app import views

# app = Flask(__name__, template_folder='templates', static_folder='static')

# app.add_url_rule('/', view_func=views.index)
# app.add_url_rule('/app/', view_func=views.app_page)
# app.add_url_rule('/app/gender/', view_func=views.genderapp, methods=['GET', 'POST'])

# if __name__ == "__main__":
#     app.run(debug=True)
