import os
import sys
import unittest
import uuid
from unittest.mock import patch

from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from Code.app import main


class FakeGoogleClient:
    def __init__(self, userinfo):
        self.userinfo = userinfo
        self.redirect_uri = None

    async def authorize_redirect(self, request, redirect_uri):
        self.redirect_uri = redirect_uri
        return RedirectResponse(
            "https://accounts.google.com/o/oauth2/v2/auth",
            status_code=302
        )

    async def authorize_access_token(self, request):
        return {"userinfo": self.userinfo}


class GoogleAuthRouteTests(unittest.TestCase):
    def test_email_registration_login_and_duplicate(self):
        with TestClient(main.app) as client:
            email = f"roadguard-{uuid.uuid4().hex}@example.com"
            registered = client.post("/auth/register", json={"name": "Road Inspector", "email": email, "password": "secure-pass-123"})
            self.assertEqual(registered.status_code, 200)
            self.assertTrue(client.get("/auth/me").json()["authenticated"])
            client.post("/auth/logout")
            self.assertEqual(client.post("/auth/login", json={"email": email, "password": "wrong-pass"}).status_code, 401)
            logged_in = client.post("/auth/login", json={"email": email, "password": "secure-pass-123"})
            self.assertEqual(logged_in.status_code, 200)
            self.assertEqual(logged_in.json()["user"]["name"], "Road Inspector")
            self.assertEqual(client.post("/auth/register", json={"name": "Other User", "email": email, "password": "secure-pass-123"}).status_code, 409)

    def test_guest_state_and_unconfigured_login(self):
        with patch.object(main, "GOOGLE_SIGN_IN_ENABLED", False), TestClient(main.app) as client:
            response = client.get("/auth/me")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {"configured": False, "authenticated": False, "user": None}
            )
            cross_origin = client.get(
                "/auth/me",
                headers={"Origin": "https://untrusted.example"}
            )
            self.assertNotIn("access-control-allow-origin", cross_origin.headers)
            self.assertEqual(client.get("/auth/google/login").status_code, 503)

    def test_callback_stores_profile_and_logout_clears_session(self):
        google = FakeGoogleClient({
            "sub": "google-user-id",
            "name": "Road Inspector",
            "email": "inspector@example.com"
        })
        with (
            patch.object(main, "GOOGLE_SIGN_IN_ENABLED", True),
            patch.object(main.oauth, "create_client", return_value=google),
            TestClient(main.app) as client
        ):
            login_response = client.get("/auth/google/login", follow_redirects=False)
            self.assertEqual(login_response.status_code, 302)
            self.assertEqual(
                login_response.headers["location"],
                "https://accounts.google.com/o/oauth2/v2/auth"
            )
            self.assertEqual(google.redirect_uri, main.GOOGLE_REDIRECT_URI)

            callback_response = client.get("/auth/google/callback?code=test&state=test")
            self.assertEqual(callback_response.status_code, 200)
            self.assertEqual(
                client.get("/auth/me").json(),
                {
                    "configured": True,
                    "authenticated": True,
                    "user": {
                        "sub": "google-user-id",
                        "name": "Road Inspector",
                        "email": "inspector@example.com"
                    }
                }
            )

            self.assertEqual(client.post("/auth/logout").json(), {"authenticated": False})
            self.assertFalse(client.get("/auth/me").json()["authenticated"])

    def test_callback_rejects_profile_without_subject(self):
        google = FakeGoogleClient({"email": "inspector@example.com"})
        with (
            patch.object(main, "GOOGLE_SIGN_IN_ENABLED", True),
            patch.object(main.oauth, "create_client", return_value=google),
            TestClient(main.app) as client
        ):
            response = client.get("/auth/google/callback?code=test&state=test")
            self.assertEqual(response.status_code, 502)
            self.assertFalse(client.get("/auth/me").json()["authenticated"])


if __name__ == "__main__":
    unittest.main()
