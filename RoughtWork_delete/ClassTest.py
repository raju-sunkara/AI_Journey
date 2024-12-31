#Class Test 
class ClassTest:
    var1:int
    var2:int
    
    def __init__(self,x,y):
        self.var1=x
        self.var2=y
    def update(self,x,y):
        print("before upate ", self.var1, self.var2)
        self.var1=x
        self.var2=y
        print("post upate ", self.var1, self.var2)
    def print(self):
        print("Variables", self.var1, self.var2)
class ClassTest1(ClassTest):
    def __init__(self,x,y):
        super().__init__(x,y)
    def print(self):
        super().print()
        print("This is from ClassTest1")
        
obj=ClassTest1(20,30)
obj.print()
obj.update(40,50)
obj.print()

