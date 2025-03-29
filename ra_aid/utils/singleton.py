"""Singleton metaclass implementation for creating singleton classes."""

class Singleton(type):
    """
    Singleton metaclass for ensuring only one instance of a class exists.
    
    Usage:
        class MyClass(metaclass=Singleton):
            pass
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            # Reinitialize existing instance if _initialize method exists
            if hasattr(cls._instances[cls], '_initialize'):
                cls._instances[cls]._initialize(*args, **kwargs)
        return cls._instances[cls]
