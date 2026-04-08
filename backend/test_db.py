from api.database import engine

def test_connection():
    try:
        with engine.connect() as connection:
            print("✅ SUCCESS: Successfully connected to the Neon database!")
    except Exception as e:
        print("❌ FAILED: Could not connect to Neon.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    test_connection()