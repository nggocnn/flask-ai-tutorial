from flaskblog import app, db
from flaskblog.models import User, Post

if __name__ == "__main__":
    with app.app_context():
        db.create_all()