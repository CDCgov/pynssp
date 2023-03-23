class NSSPContainer:
    def __init__(self, value):
        self.value = value


class APIGraph:
  def __init__(self, path, response):
    self.path = path
    self.response = response

  def __str__(self):
    return f"{self.path}"