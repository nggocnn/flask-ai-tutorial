from flaskblog import db, create_app
from flaskblog.models import User, Post, OriginImage, AdvImage

app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        