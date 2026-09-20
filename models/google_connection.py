# """
# Google OAuth connection model.

# Stores one Google Workspace connection per platform user.
# OAuth tokens are encrypted before being stored.
# """

# from datetime import datetime

# from sqlalchemy import (
#     Column,
#     DateTime,
#     ForeignKey,
#     Integer,
#     String,
#     UniqueConstraint,
# )
# from sqlalchemy.orm import relationship

# from core.database import Base


# class GoogleConnection(Base):
#     __tablename__ = "google_connections"

#     id = Column(Integer, primary_key=True, index=True)

#     user_id = Column(
#         Integer,
#         ForeignKey("users.id", ondelete="CASCADE"),
#         nullable=False,
#         index=True,
#     )

#     provider = Column(
#         String(50),
#         nullable=False,
#         default="google",
#     )

#     email = Column(
#         String(255),
#         nullable=True,
#     )

#     access_token = Column(
#         String(4096),
#         nullable=False,
#     )

#     refresh_token = Column(
#         String(4096),
#         nullable=True,
#     )

#     token_expiry = Column(
#         DateTime(timezone=True),
#         nullable=True,
#     )

#     scopes = Column(
#         String(4096),
#         nullable=True,
#     )

#     created_at = Column(
#         DateTime(timezone=True),
#         default=datetime.utcnow,
#         nullable=False,
#     )

#     updated_at = Column(
#         DateTime(timezone=True),
#         default=datetime.utcnow,
#         onupdate=datetime.utcnow,
#         nullable=False,
#     )

#     revoked_at = Column(
#         DateTime(timezone=True),
#         nullable=True,
#     )

#     user = relationship(
#         "User",
#         back_populates="google_connections",
#     )

#     __table_args__ = (
#         UniqueConstraint(
#             "user_id",
#             "provider",
#             name="uq_google_connections_user_provider",
#         ),
#     )