from app.core.config import get_settings

settings = get_settings()

print(f'App_name: {settings.app_name}')
print(f'API_KEY: {settings.groq_api_key}')