class ControlledDict(dict):
    __readonly = False

    def block_setitem(self):
        """Allow or deny modifying dictionary"""
        self.__readonly = True

    def allow_setitem(self):
        """Allow or deny modifying dictionary"""
        self.__readonly = False

    def __setitem__(self, key, value):

        if self.__readonly:
            raise(TypeError, "__setitem__ is not supported")
        return dict.__setitem__(self, key, value)

    def __delitem__(self, key):

        if self.__readonly:
            raise(TypeError, "__delitem__ is not supported")
        return dict.__delitem__(self, key)
