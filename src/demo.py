# this file can be deleted and serves demo purposes only

def say_hello(name: str) -> str:
    """
    This function returns a string that contains a greeting for a person.
    It accepts a name as a string variable as input.

    :param name: a name as a string variable
    :return: a string variable that contains a greeting
    """

    return f'hello {name}'


if __name__ == '__main__':
    
    greeting = say_hello(name='louis')
    print(greeting)