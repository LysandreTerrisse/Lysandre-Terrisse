# -*- coding: utf-8 -*-
from turtle import *
from math import sqrt, acos, pi, tan, sin, ceil
from time import sleep
from random import randint

try:
    reset()
except Terminator:
    pass

"""
Les distances ne sont pas mesurées en pixels pour que le disque de Poincaré
ait un rayon de 1
"""

class Polygon:
    def __init__(self, points=[], neighbours=[], sides_with_children=[], center=[], state=0):
        self.points=points + []
        self.neighbours=neighbours + []
        self.sides_with_children=sides_with_children
        self.center=center
        self.state=state
 
    def getPoints(self):
        return self.points
    
    def getPoint(self, n_point):
        return self.points[n_point]
    
    def getSide(self, n_side):
        p1 = self.points[n_side]
        p2 = self.points[0 if n_side+1==n_sides else n_side+1]
        return (p1, p2)
    
    def getNeighbours(self):
        return self.neighbours
    
    def getNeighbour(self, n_side):
        neighbours = self.getNeighbours()
        if neighbours:
            return neighbours[n_side]
    
    def setPoints(self, points):
        self.points=points
    
    def addNeighbour(self, neighbour):
        self.neighbours.append(neighbour)
    
    def addPoint(self, point):
        self.points.append(point)
        
    def inversePointsAndCenter(self, n_side):
        """
        Renvoie l'inverse du polygone selon le côté d'indice n_side.
        
        p1 et p2 sont sur une même droite hyperbolique. On cherche un
        troisième point sur le cercle correspondant pour calculer le centre
        du cercle. Pour cela, on prend l'inverse de p1.
        
        L'inverse par rapport à (0; 0) d'un point compris sur une droite
        hyperbolique est aussi compris sur le cercle correspondant à cette droite.
        
        Il se peut que (0, 0), p1, et p2 soient alignés. Dans ce cas, on fait
        une symétrie axiale.
        """
        (p1, p2) = self.getSide(n_side)
        inversed_polygon = Polygon()
        inversed_points = [0]*n_sides
        self_center = self.getCenter()
        self_points = self.getPoints()
        
        if aligned((0, 0), p1, p2):
            inversed_polygon.setCenter(axialSymmetry(p1, p2, self_center))
            for i in range(n_sides):
                inversed_points[i] = axialSymmetry(p1, p2, self_points[i])
            
        else:
            p3 = inverse((0, 0), p1, 1)
            p4 = center(p1, p2, p3)
            r = distance(p4, p1)
            
            inversed_polygon.setCenter(inverse(p4, self_center, r))
            for i in range(n_sides):
                inversed_points[i] = inverse(p4, self_points[i], r)
        
        """
        À cause de la réflexion, les points ne sont pas dans le sens
        trigonométrique. On doit régler cela en faisant :
        """
        
        inversed_points.reverse()
        border = (n_sides-1)-(n_side+1)
        inversed_polygon.setPoints(inversed_points[border:] + inversed_points[0:border])
        
        return inversed_polygon
    
    def inversePoints(self, n_side):
        """N'inverse que les points."""
        (p1, p2) = self.getSide(n_side)
        inversed_points = [0]*n_sides
        self_points = self.getPoints()
        
        if aligned((0, 0), p1, p2):
            for i in range(n_sides):
                inversed_points[i] = axialSymmetry(p1, p2, self_points[i])
            
        else:
            p3 = inverse((0, 0), p1, 1)
            p4 = center(p1, p2, p3)
            r = distance(p4, p1)
            
            for i in range(n_sides):
                inversed_points[i] = inverse(p4, self_points[i], r)
        
        inversed_points.reverse()
        border = (n_sides-1)-(n_side+1)
        return inversed_points[border:] + inversed_points[0:border]
    
    def draw(self, n_first_side=1):
        """Dessine le polygone
        On dessine à partir de la face n_first_side.
        On ne dessine pas les polygones d'état 3 (sauf dans le mode Test)car ce
        sont des culs-de-sac. Ils ne sont utilisés que dans pointedSide et
        draw."""
        points = self.getPoints()
        state = self.getState()
        

        point = points[n_first_side][0]*zoom, points[n_first_side][1]*zoom
        if not corresponds(position(), point):
            up()
            goto(point)
            down()
        
        if state!=0:
            if state!=1:
                color("red")
            begin_fill()
        
        for n_side in range(n_first_side+1, n_sides):
            (px, py) = points[n_side]
            goto(px*zoom, py*zoom)
        goto(points[0][0]*zoom, points[0][1]*zoom)
        
        if state!=0:
            end_fill()
            if state!=1:
                color("black")
        
       

    def getChildren(self):
        children = []
        neighbours = self.getNeighbours()
        for n_side_with_children in self.getSidesWithChildren():
            children.append(neighbours[n_side_with_children])
        return children
    
    def setCenter(self, center):
        self.center = center
    
    def getCenter(self):
        return self.center
    
    def setSidesWithChildren(self, sides_with_children):
        self.sides_with_children = sides_with_children
        
    def getSidesWithChildren(self):
        return self.sides_with_children
    
    def calculateTree(self):
        neighbours = self.getNeighbours()
        for n_side_with_children in self.getSidesWithChildren():
            child = neighbours[n_side_with_children]
            if child:
                child.setPoints(self.inversePoints(n_side_with_children))
                child.calculateTree()
        
    def move(self, p1):
        """Déplace le polygone vers p1
        
        Pour que le centre p1 du polygone ne soit pas en p0 (0, 0), on trace
        une droite passant par p0 et p1, puis sa perpendiculaire d passant par
        p1. A est un point en commun entre d et l'horizon. On inverse
        le polygone selon le cercle de centre p1' (ou p2) et de rayon
        distance(p2, A) (ou r2)
    		
    	On peut déterminer r2 à partir de distance(p2, A) (ou side_1), et de
        distance(p1, p2) (ou side_2). side_1 est le sinus de l'arcos de
        distance(p0, p1).
        """
        p0 = (0, 0)
        r = distance(p0, p1)
        if r<1:
            if p1!=p0:
                p2 = inverse(p0, p1, 1)
                
                side_1 = sin(acos(r))
                side_2 = distance(p1, p2)
                r2 = sqrt(side_1**2 + side_2**2)
                
                
                points = []
                for point in center_polygon.getPoints():
                    points.append(inverse(p2, point, r2))
                
                """
                À cause de la réflection, on doit inverser
                """
                points.reverse()
                
                self.setPoints(points)
            
            else:
                self.setPoints(center_polygon.getPoints())
            
            
            
    def propagation(self):
        """Répercute les changement de self vers first_polygon par récursion"""
        if self != first_polygon:
            polygon_to_change = self.getNeighbour(0)
            inversed_points = self.inversePoints(0)
            """Pour l'instant, inverse_polygon montre son côté 0 à
            changed_polygon. On veut qu'il montre son côté n_side, où
            n_side représente l'indice de changed_polygon dans
            polygon_to_change.getNeighbours()"""
            
            n_side = polygon_to_change.getNeighbours().index(self)
            border = n_sides-n_side
            inversed_points = inversed_points[border:] + inversed_points[0:border]
            
            polygon_to_change.setPoints(inversed_points)
            
            polygon_to_change.propagation()
    
    def setState(self, state):
        self.state=state
    
    def getState(self):
        return self.state

    def erase(self, neighbour):
        """Remplace neighbour dans self.neighbours par None, et l'enlève de la
        table des enfants"""
        neighbours = self.getNeighbours()
        n_side = neighbours.index(neighbour)
        neighbours[n_side] = None
        if n_side in self.sides_with_children:
            self.sides_with_children.remove(n_side)
    
def strictlyBigger(x):
    """Retourne le plus petit nombre naturel strictement plus grand que x"""
    x2 = ceil(x)
    return x2+1 if equals(x, x2) else x2

def distance(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
def inverse(p1, p2, r):
    """
    Retourne p, l'inverse du point p2 par rapport à la droite hyperbolique
    qui a comme centre p1 et comme rayon r.
    
    distance(p1, p) doit être r**2 / distance(p1, p2)
    et
    coeff * distance(p1, p2) doit donner distance(p1, p)
    donc
    coeff = distance(p1, p) / distance(p1, p2)
    <=> coeff = r**2 / distance(p1, p2)**2
    <=> coeff = ( r/distance(p1, p2) )**2
    """
    p1p2 = (p2[0]-p1[0], p2[1]-p1[1])
    
    coeff = r**2/(p1p2[0]**2 + p1p2[1]**2)
    return (p1[0] + p1p2[0]*coeff, p1[1] + p1p2[1]*coeff)

def aligned(p1, p2, p3):
    """Retourne si p1, p2, et p3 sont alignés entre eux
    
    Il existe un réel t tel que t*p1p2 = p1p3
    <=>
        t*x_p1p2 = x_p1p3
        t*y_p1p2 = y_p1p3
    
    """
    (x_p1p2, y_p1p2) = (p2[0] - p1[0], p2[1] - p1[1])
    (x_p1p3, y_p1p3) = (p3[0] - p1[0], p3[1] - p1[1])
    
    if equals(x_p1p2, 0):
        return equals(x_p1p3, 0)
    
    if equals(y_p1p2, 0):
        return equals(y_p1p3, 0)
    
    return equals(x_p1p3/x_p1p2, y_p1p3/y_p1p2)
    
def orthogonalProjection(p1, p2, p3):
    """Retourne p4, le projeté orthogonal de p3 sur la droite p1p2
    
    p4 est compris dans la droite p1p2 :
    <=> p4 = (x_p1 + x_p1p2*t ; y_p1 + y_p1p2*t)
    <=> p3p4 = (x_p1-xp3 + x_p1p2*t ; y_p1-yp3 + y_p1p2*t)
    
    p3p4.p1p2 = 0
    <=> (x_p1-x_p3 + x_p1p2*t)*x_p1p2 + (y_p1-y_p3 + y_p1p2*t)*y_p1p2 = 0
    <=> (x_p1-x_p3)*x_p1p2 + x_p1p2²*t + (y_p1-y_p3)*y_p1p2 + y_p1p2²*t = 0
    <=> ((x_p1-x_p3)*x_p1p2 + (y_p1-y_p3)*y_p1p2) + (x_p1p2² + y_p1p2²)*t = 0
    <=> t = - ((x_p1-x_p3)*x_p1p2 + (y_p1-y_p3)*y_p1p2) / (x_p1p2² + y_p1p2²)
    <=> t = ((x_p3-x_p1)*x_p1p2 + (y_p3-y_p1)*y_p1p2) / (x_p1p2² + y_p1p2²)
    """
    (x_p1, y_p1) = p1
    (x_p2, y_p2) = p2
    (x_p3, y_p3) = p3
    (x_p1p2, y_p1p2) = (x_p2 - x_p1, y_p2 - y_p1)
    
    t = ((x_p3-x_p1)*x_p1p2 + (y_p3-y_p1)*y_p1p2) / (x_p1p2**2 + y_p1p2**2)
    return (x_p1 + x_p1p2*t, y_p1 + y_p1p2*t)
    
def axialSymmetry(p1, p2, p3):
    """Retourne p5, la symmétrie de p3 par la droite p1p2"""
    (x_p3, y_p3) = p3
    (x_p4, y_p4) = orthogonalProjection(p1, p2, p3)
    (x_p3p4, y_p3p4) = (x_p4 - x_p3, y_p4 - y_p3)
    return (x_p3+2*x_p3p4, y_p3+2*y_p3p4)

def drawPoint(p):
    up()
    goto(p[0]*zoom, p[1]*zoom)
    down()
    width(8)
    forward(1)
    width(thickness)

def drawCircle(p, r):
    up()
    goto(p[0]*zoom, (p[1]-r)*zoom)
    setheading(0)
    down()
    circle(r*zoom)
    up()

def angle360(v):
    """
     Calcule l'angle de v dans [0; 360[ :
        a.b = a * b * cos(a, b)
        Donc angle(a, b)(de 0 à 180) = acos(a.b / a*b)
        Donc angle((0, 1), v)(de 0 à 180) = acos(1*vx+0*vy / 1*v)
        Donc angle(v)(de 0 à 180) = acos(vx/v)
        
        Pour aller jusqu'à 360, on soustrait l'angle à 360 si vy < 0
    """
    len_v = distance((0, 0), v)
    if v[1]<0:
        return 360 - acos(v[0]/len_v)/pi * 180
    else:
        return acos(v[0]/len_v)/pi * 180
    

def bisector(p1, p2):
    """Retourne le coefficient directeur et l'ordonnée à l'origine de la
    médiatrice de v
    
    Médiatrice (bisector en anglais) : Droite perpendiculaire passant au centre
    d'un segment
    """
    p1p2 = (p2[0]-p1[0], p2[1]-p1[1])
    a = -p1p2[0] / p1p2[1]
    b = (p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2) / (2*p1p2[1])
    return (a, b)

def center(p1, p2, p3):
    """
    Retourne p, le centre du cercle passant par p1, p2, et p3
    
    Il se peut que bis_p1p2 soit parallèle à l'axe des ordonnées. dans ce
    cas, il faut trouver l'ordonnée de bis_p2p3 quand l'abscisse est égale au
    centre de p1p2. Vice versa dans le cas de bis_p2p3.
    """
    
    p1p2 = (p2[0]-p1[0], p2[1]-p1[1])
    p2p3 = (p3[0]-p2[0], p3[1]-p2[1])
    
    if equals(p1p2[1], 0):
        bis_p2p3 = bisector(p2, p3)
        x = (p1[0] + p2[0])/2
        y = bis_p2p3[0]*x + bis_p2p3[1]
        
    elif equals(p2p3[1], 0):
        bis_p1p2 = bisector(p1, p2)
        x = (p2[0] + p3[0])/2
        y = bis_p1p2[0]*x + bis_p1p2[1]
        
    else:
        bis_p1p2 = bisector(p1, p2)
        bis_p2p3 = bisector(p2, p3)
        
        x = (bis_p1p2[1] - bis_p2p3[1]) / (bis_p2p3[0] - bis_p1p2[0])
        y = bis_p1p2[0]*x + bis_p1p2[1]
    
    return (x, y)

def equals(a, b):
    """Compare des nombres flottants"""
    return -precision < a-b < precision

def corresponds(p1, p2):
    """Compare des points flottants"""
    return equals(p1[0], p2[0]) and equals(p1[1], p2[1])
    
def getBorderPolygons(t_all):
    """Retourne tous les polygones qui n'ont pas encore de voisins définis"""
    border_polygons = []
    for polygon in t_all:
        if not polygon.getNeighbours():
            border_polygons.append(polygon)
    
    return border_polygons
    
def getCorrespondingPolygon(t_all, polygon):
    """Retourne quel polygone contient les deux points"""
    center_of_polygon = polygon.getCenter()
    
    for potential_polygon in t_all:
        if corresponds(potential_polygon.getCenter(), center_of_polygon):
            return potential_polygon

def getCenterPolygon(start_angle=0):
    """
    n_sides est le nombre de côtés des polygones
    n_per_corner est le nombre de polygones qui se rejoignent par coin
    
        En géométrie hyperbolique, les polygones peuvent paver le disque seulement
    quand ils ont une taille bien précise. Si A est un polygone au centre du
    disque, alors la distance euclidienne entre un de ses coins et le centre du
    disque est d situé ci dessous
    """
    
    angle = 360/n_sides
    d = sqrt((tan(pi/2 - pi/n_per_corner) - tan(pi/n_sides)) / (tan(pi/2 - pi/n_per_corner) + tan(pi/n_sides)))
    first_polygon = Polygon()
    first_polygon.setCenter((0, 0))
    
    up()
    for n_side in range(n_sides):
        goto(0, 0)
        setheading((n_side)*angle + start_angle)
        forward(d*zoom)
        (x, y) = position()
        first_polygon.addPoint((x/zoom, y/zoom))
    
    return first_polygon
    
def calculateTreeForFirstTime():
    """Calcule l'arbre de first_polygon à partir de ses cordonnées"""
    t = [first_polygon]
    for _ in range(recursivity):
        for polygon in getBorderPolygons(t):
            sides_with_children = []
            for n_side in range(n_sides):
                child = polygon.inversePointsAndCenter(n_side)
                corresponding_polygon = getCorrespondingPolygon(t, child)
                
                if corresponding_polygon:
                    polygon.addNeighbour(corresponding_polygon)
                else:
                    polygon.addNeighbour(child)
                    sides_with_children.append(n_side)
            
            polygon.setSidesWithChildren(sides_with_children)
            t = t + polygon.getChildren()
    
    
    for polygon in getBorderPolygons(t):
        for n_side in range(n_sides):
            neighbour = polygon.inversePointsAndCenter(n_side)
            corresponding_polygon = getCorrespondingPolygon(t, neighbour)
            """Peut être amélioré"""
            polygon.addNeighbour(corresponding_polygon)
    
    t2 = t + []
    first_loop = True
    while t != t2 or first_loop:
        first_loop = False
        t = t2 + []
            
        for polygon in t:
            n_neighbours = 0
            for neighbour in polygon.getNeighbours():
                if neighbour:
                    n_neighbours+=1
            if n_neighbours<2:
                for neighbour in polygon.getNeighbours():
                    if neighbour:
                        neighbour.erase(polygon)
                t2.remove(polygon)
    return t

def detectMousePosition(event):
    """Met à jour les coordonnées de la souris"""
    global mouse
    screen_x, screen_y = canvas.winfo_width(), canvas.winfo_height()
    mouse = ((event.x - screen_x/2)/zoom, (screen_y/2 - event.y)/zoom)

def detectClick(event):
    """Pour savoir quel curseur bouger quand on le déplace, il faut savoir sur
    lequel on a appuyé. 0 correspond à rien, 1 au curseur de n_sides, et 2 au
    curseur de n_per_corner."""
    global cursor
    detectMousePosition(event)
    
    n_sides_options = n_sides_max - n_sides_min + 1
    n_cursor_options = n_per_corner_max - n_per_corner_min + 1
    n_sides_cursor = ((n_sides-n_sides_min)*2/n_sides_options - 1, 1)
    n_per_corner_cursor = ((n_per_corner-n_per_corner_min)*2/n_cursor_options - 1, -1)
    
    if distance(n_sides_cursor, mouse) < 0.1:
        cursor = 1
    elif distance(n_per_corner_cursor, mouse) < 0.1:
         cursor = 2
    else:
        cursor = 0
    
def detectMouseDrag(event):
    global drag
    detectMousePosition(event)
    drag = 1

def detectEscapePressed():
    global escape
    if escape and t_all:
        escape = False
    else:
        escape = True

def drawTiling():
    """Dessine le pavage"""
    reset()
    width(thickness)
    hideturtle()
    drawCircle((0, 0), 1)
    
    first_polygon.draw(0)
    for polygon in t_all[1:]:
        if polygon:
            polygon.draw()
        
    if escape:
        drawCursor(1, n_sides_min, n_sides_max, n_sides)
        drawCursor(-1, n_per_corner_min, n_per_corner_max, n_per_corner)
        if not t_all:
            goto((0, 0))
            color("white")
            write("La récursivité est trop faible. Le pavage ne contient que des culs-de-sac", align="center", font=personalized_font)
    
    update()

def moveTowardsSide(n_side):
    """Déplace le polygone n_side vers le centre"""
    global polygon_at_center
    
    """On regarde le point de départ de polygon_side"""
    polygon_side = polygon_at_center.getNeighbour(n_side)
    p_beginning = getCenterOfSide(n_side)
    
    """On cherche le côté en commun entre polygon_side et le polygone
    central"""
    n_side_common = polygon_side.getNeighbours().index(polygon_at_center)
    n_side_opposite = (n_sides-1) - (n_side+1) if n_side!=n_sides else n_sides
    diff_side = n_side_opposite - n_side_common
        
    for n_step in range(1, n_steps_per_move_animation):
        if escape:
            """On coupe l'animation"""
            return
        
        """polygon_side fait un petit pas de p_beginning à (0, 0)"""
        coeff = (n_steps_per_move_animation-n_step)/n_steps_per_move_animation
        p = (coeff*p_beginning[0], coeff*p_beginning[1])
        polygon_side.move(p)
        
        """Ici, polygone_side montre son côté n_side_opposite au polygone
        central. On voudrait qu'il montre le côté en commun avec le polygone
        central"""
        points = polygon_side.getPoints()
        points = points[diff_side:] + points[0:diff_side]
        polygon_side.setPoints(points)
        
        polygon_side.propagation()
        first_polygon.calculateTree()
        drawTiling()
    
    """On fait en sorte que la prochaine fois qu'on déplace polygone_side, il
    montre bien vers (0, 0) un côté et pas un coin. Pour cela, on tourne
    center_polygon pour qu'il soit identique au futur polygone au centre.
    Il faut obtenir un angle plus précis du futur polygone, c'est à quoi sert
    tmp_polygon (qui est comme le futur polygone mais encore plus centré)."""
    if escape:
        return
    tmp_polygon = getCenterPolygon()
    tmp_polygon.move((precision2*p_beginning[0], precision2*p_beginning[1]))
    """Peut être amélioré"""
    points = tmp_polygon.getPoints()
    points = points[diff_side:] + points[0:diff_side]
    tmp_polygon.setPoints(points)
        
    new_current_angle = angle360(tmp_polygon.getPoint(0))
    """On arrondit l'angle"""
    new_current_angle = round(new_current_angle*2*n_sides/360)*360/(2*n_sides)
    
    """On affiche la dernière image du mouvement"""
    center_polygon.setPoints(getCenterPolygon(new_current_angle).getPoints())
    polygon_side.move((0, 0))
    polygon_side.propagation()
    first_polygon.calculateTree()
    drawTiling()
    polygon_at_center = polygon_at_center.getNeighbour(n_side)
    
def pointedSide(mouse):
    """Retourne le côté du polygone central qui est le plus proche de la
    souris. On fait cela en comparant la souris et le centre de chaque polygone
    voisin."""
    neighbours = polygon_at_center.getNeighbours()
    
    (p1, p2) = polygon_at_center.getSide(0)
    side_center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    shortest_distance = distance(mouse, side_center)
    closest_side = 0
    
    for n_side in range(1, n_sides):
        (p1, p2) = polygon_at_center.getSide(n_side)
        side_center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        dist = distance(mouse, side_center)
        if shortest_distance > dist:
            shortest_distance = dist
            closest_side = n_side
    
    return closest_side
    
def chooseRandomFruit():
    """Place sur le pavage un fruit à un emplacement disponible"""
    random_polygon = t_all[randint(0, len(t_all)-1)]
    if random_polygon.getState()!=0:
        chooseRandomFruit()
    else:
        random_polygon.setState(2)

def drawCursor(y, n_min, n_max, n):
    """Dessine un curseur de longueur 2 et d'ordonnée y, qui
    commence par la valeur n_min, et qui finit par la valeur n_max, avec comme
    valeur actuelle n"""
    up()
    width(15)
    color(0.9, 0.9, 0.9)
    
    goto(-zoom, y*zoom)
    down()
    goto(zoom, y*zoom)
    up()
    
    width(1)
    color("blue")
    n_options = n_max - n_min + 1
    for i in range(n_options+1):
        x = i*zoom*2/n_options - zoom
        goto(x, y*zoom - 7.5)
        down()
        goto(x, y*zoom + 7.5)
        up()
    
    color("black", "white")
    up()
    cursor = ((n-n_min)*2/n_options - 1, y)
    goto(cursor[0]*zoom, y*zoom)
    begin_fill()
    drawCircle(cursor, 0.1)
    end_fill()
    
def changeOptions():
    """Modifie n_sides, n_per_corner, n_sides_min, et n_per_corner_min selon
    le choix du joueur"""
    global escape, n_sides, n_per_corner, drag, first_polygon, center_polygon, polygon_at_center, t_all
    global n_sides_min, n_per_corner_min, n_sides_max, n_per_corner_max
    
    has_to_refresh = False
    if cursor==1:
        tmp = int((mouse[0]+1)*(n_sides_max - n_sides_min + 1)/2) + n_sides_min
        if n_sides!=tmp:
            n_sides = tmp
            n_per_corner_min = strictlyBigger(1/(1/2 - 1/n_sides))
            has_to_refresh = True
            
    else:
        tmp = int((mouse[0]+1)*(n_per_corner_max - n_per_corner_min + 1)/2) + n_per_corner_min
        if n_per_corner!=tmp:
            n_per_corner = tmp
            n_sides_min = strictlyBigger(1/(1/2 - 1/n_per_corner))
            has_to_refresh = True
    
    if has_to_refresh:
        first_polygon = getCenterPolygon()
        center_polygon = getCenterPolygon()
        polygon_at_center = first_polygon
        t_all = calculateTreeForFirstTime()
        drawTiling()
    
def activateMenu():
    """Le menu peut être activé en appuyant sur Échap"""
    global escape, n_sides, n_per_corner, drag, first_polygon, center_polygon, polygon_at_center, t_all
    drawTiling()
    bgcolor("gray")
    
    while escape:
        drag=False
        update()
        if drag and cursor!=0 and -1<=mouse[0]<1:
            changeOptions()
            
        
    drawTiling()
    bgcolor("white")
    escape = False
    

def activateTest():
    """Le test peut être activé en modifiant la variable test
    Dans le test, le premier polygone a comme centre celui de la souris, et
    les polygones de type 3 sont visibles"""
    global mouse
    mouse = (0, 0)
    while True:
        if escape:
            break
        
        mouse2 = mouse
        first_polygon.move(mouse2)
        first_polygon.propagation()
        first_polygon.calculateTree()
        drawTiling()

def activateSnake():
    """Active le jeu"""
    global game_launched
    snake_table = [first_polygon]
    first_polygon.setState(1)
    chooseRandomFruit()
    is_alive = True
    
    if not mouse:
        hideturtle()
        goto(0, 0)
        write("Bougez la souris pour commencer", align="center", font=personalized_font)
        while not mouse:
            update()
    
    drawTiling()
    sleep(1)
    
    while is_alive:
        update()
        mouse2 = mouse
        pointed_side = pointedSide(mouse2)
        
        polygon_side = polygon_at_center.getNeighbour(pointed_side)
        state = polygon_side.getState() if polygon_side else None
        
        if state==1 or state==None:
            is_alive = False
            
        else:
            moveTowardsSide(pointed_side)
            if escape:
                break
            
            if state==0:
                snake_table.pop(0).setState(0)
            else:
                try:
                    chooseRandomFruit()
                except RecursionError:
                    break
                
            snake_table.append(polygon_at_center)
            polygon_at_center.setState(1)
            drawTiling()
            sleep(seconds_to_pause)
    
    if not escape:
        reset()
        write("Jeu gagné !" if is_alive else "Game Over", align="center", font=personalized_font)
        game_launched = False

def getCenterOfSide(n_side):
    """Retourne le centre du polygone n_side de center_polygon"""
    (p1, p2) = center_polygon.getSide(n_side)
    
    p3 = inverse((0, 0), p1, 1)
    p4 = center(p1, p2, p3)
    r = distance(p4, p1)
    
    return inverse(p4, (0, 0), r)
    

tracer(0, 0)#Pour ne pas rafraîchir automatiquement
bgcolor("white")
hideturtle()

"""
- Variables pouvant être modifiées par l'utilisateur -
"""
zoom = 300
thickness = 2
recursivity = 4
n_steps_per_move_animation = 20
seconds_to_pause = 0.5
test = False





personalized_font = ("Arial", 20)
precision = 0.00001
precision2 = 0.01#Utilisé pour obtenir l'angle du nouveau polygone central
n_sides = 4
n_per_corner = 5
n_sides_min = 4
n_per_corner_min = 5
n_sides_max = 10
n_per_corner_max = 10

mouse = None
drag = False
cursor = 0
canvas = getcanvas()
canvas.bind('<Motion>', detectMousePosition)
canvas.bind("<Button-1>", detectClick)
canvas.bind("<B1-Motion>", detectMouseDrag)
escape = False
onkeypress(detectEscapePressed, 'Escape')
listen()


game_launched = True

try:
	while game_launched:
		if escape:
			activateMenu()
		
		first_polygon = getCenterPolygon()
		center_polygon = getCenterPolygon()
		polygon_at_center = first_polygon
		t_all = calculateTreeForFirstTime()
		if test:
			activateTest()
		else:
			activateSnake()
	
	exitonclick()
	
except Terminator:
	pass