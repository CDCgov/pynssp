class NSSPContainer:
  """A NSSPContainer Class to store a value or an object

    An object of class NSSPContainer stores a value or an object
    The NSSPContainer class encapsulates a value or an object
  """

  def __init__(self, value):
    """Initializes an NSSPContainer class

    :param value: value to store
    """
    self.value = value


class APIGraph:
  """An class to store an API graph"""

  def __init__(self, path, response):
    """Initializes an APIGraph class

    :param path: a string representing the location of a graph
    :param response: an object of class response
    """
    self.path = path
    self.response = response


  def __str__(self):
    """Print an API object"""
    return f"{self.path}"


  def show(self):
    """A method to display an APIGraph object"""
    from PIL import Image
    img = Image.open(self.path)
    img.show()


  def plot(self):
    """A method to plot an APIGraph object"""
    from skimage.io import imread, imshow
    img = imread(self.path)
    imshow(img)
