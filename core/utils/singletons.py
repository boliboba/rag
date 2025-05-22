import functools


def lazy_singleton(init_func=None):
    
    def decorator(func):
        # Инстанс для хранения объекта
        instance = {'value': None}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Если объект уже создан и нет аргументов - возвращаем его
            if instance['value'] is not None and not (args or kwargs):
                return instance['value']
            
            # Иначе создаем объект и сохраняем
            instance['value'] = func(*args, **kwargs)
            return instance['value']
        
        # Функция для сброса кэша
        def reset():
            instance['value'] = None
        
        # Добавляем метод сброса кэша
        wrapper.reset = reset
        
        return wrapper
    
    # Позволяет использовать как с аргументами, так и без
    if init_func is not None:
        return decorator(init_func)
    return decorator 