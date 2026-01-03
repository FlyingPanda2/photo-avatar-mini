import replicate
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
from functools import lru_cache

load_dotenv()

app = Flask(__name__)

# Разрешить CORS для всех источников (для GitHub Pages и Telegram Mini App)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ваш API ключ Replicate
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')

# Кэш для сгенерированных изображений (в памяти)
image_cache = {}

# Промпты для разных стилей
STYLE_PROMPTS = {
    'anime': 'anime illustration, detailed eyes, vibrant colors, manga style, professional artwork, high quality, 4k',
    'pixel': 'pixel art, 8-bit, 16-bit retro game style, isometric, colorful, detailed, video game aesthetic',
    'vintage': 'vintage polaroid photo, 1970s, film grain, warm colors, nostalgic, faded, aged photo, retro'
}

@app.route('/generate', methods=['POST'])
def generate():
    """
    Генерирует изображение через Replicate API
    Параметры:
    - style: 'anime', 'pixel' или 'vintage'
    - prompt: опциональный дополнительный промпт
    """
    try:
        if not REPLICATE_API_TOKEN:
            return jsonify({
                'error': 'REPLICATE_API_TOKEN не установлен. Добавьте в .env файл'
            }), 400

        data = request.json
        style = data.get('style', 'anime')
        user_prompt = data.get('prompt', '')

        # Проверяем кэш
        cache_key = f"{style}_{user_prompt}"
        if cache_key in image_cache:
            return jsonify({
                'url': image_cache[cache_key],
                'cached': True
            })

        # Формируем полный промпт
        base_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS['anime'])
        full_prompt = f"{base_prompt}. {user_prompt}" if user_prompt else base_prompt

        print(f"Генерирую изображение: {full_prompt[:80]}...")

        # Используем Replicate API для генерации
        # Используем Stable Diffusion XL для лучшего качества
        output = replicate.run(
            "stability-ai/stable-diffusion-3",
            input={
                "prompt": full_prompt,
                "image_dimensions": "512x512",
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
            }
        )

        if output and len(output) > 0:
            image_url = output[0] if isinstance(output, list) else output
            
            # Сохраняем в кэш
            image_cache[cache_key] = image_url
            
            return jsonify({
                'url': image_url,
                'cached': False,
                'style': style
            })
        else:
            return jsonify({
                'error': 'Не удалось сгенерировать изображение'
            }), 500

    except Exception as e:
        print(f"Ошибка при генерации: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Проверка статуса сервера"""
    return jsonify({
        'status': 'ok',
        'api_configured': bool(REPLICATE_API_TOKEN),
        'cache_size': len(image_cache)
    })

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Очистка кэша"""
    global image_cache
    size = len(image_cache)
    image_cache = {}
    return jsonify({
        'message': f'Кэш очищен ({size} элементов удалено)'
    })

if __name__ == '__main__':
    if not REPLICATE_API_TOKEN:
        print("⚠️  ВАЖНО: Установите переменную окружения REPLICATE_API_TOKEN")
        print("Инструкция: https://replicate.com/account/api-tokens")
    
    # Для локального запуска
    port = int(os.getenv('PORT', 5000))
    app.run(debug=os.getenv('FLASK_ENV') == 'development', host='0.0.0.0', port=port)
