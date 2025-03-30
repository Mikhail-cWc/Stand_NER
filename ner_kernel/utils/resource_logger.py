import time
import psutil
import functools
from typing import Callable
from ..logger import logger


def log_resources(func: Callable):
    """
    Декоратор для логирования времени выполнения и использования памяти
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Получаем процесс
        process = psutil.Process()

        # Замеряем начальное время и память
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # в МБ

        # Выполняем функцию
        result = func(*args, **kwargs)

        # Замеряем конечное время и память
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # в МБ

        # Вычисляем разницу
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        # Логируем результаты
        logger.info(
            f"Функция {func.__name__}:\n"
            f"  Время выполнения: {execution_time:.4f} сек\n"
            f"  Использовано памяти: {memory_used:.2f} МБ"
        )

        return result
    return wrapper
