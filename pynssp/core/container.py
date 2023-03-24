class NSSPContainer:
    def __init__(self, value):
        self.value = value


class APIGraph:
  def __init__(self, path, response):
    self.path = path
    self.response = response

  def __str__(self):
    return f"{self.path}"

  def show(self):
     from PIL import Image
     img = Image.open(self.path)
     img.show()