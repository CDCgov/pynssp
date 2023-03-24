"""
A NSSPContainer Class to store a value or an object
@description: An object of class NSSPContainer stores a value or an object
@details: The NSSPContainer class is used to encapsulate a value or an object
"""
class NSSPContainer:
    """
    Initializes an NSSPContainer class
    @param value: value to store
    """
    def __init__(self, value):
        self.value = value


"""
An class to store an API graph
"""
class APIGraph:
  """
  Initializes an APIGraph class
  @param path: a string representing the location of a graph
  @param response: an object of class response
  """
  def __init__(self, path, response):
    self.path = path
    self.response = response

  """
  Print an API object
  """
  def __str__(self):
    return f"{self.path}"

  """
  A method to display a graph
  """
  def show(self):
     from PIL import Image
     img = Image.open(self.path)
     img.show()