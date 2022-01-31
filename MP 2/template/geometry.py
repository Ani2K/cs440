# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from numpy.core.numeric import isclose
from alien import Alien

def dist_line_line(line1X1, line1Y1, line1X2, line1Y2, line2X1, line2Y1, line2X2, line2Y2):
    changeX1 = line1X2 - line1X1
    changeY1 = line1Y2 - line1Y1
    changeX2 = line2X2 - line2X1
    changeY2 = line2Y2 - line2Y1
    offset = changeX2 * changeY1 - changeY2 * changeX1
    if (offset != 0):
        a = (changeX1 * (line2Y1 - line1Y1) + changeY1 * (line1X1 - line2X1)) / offset
        b = (changeX2 * (line1Y1 - line2Y1) + changeY2 * (line2X1 - line1X1)) / (-offset)
        if ((0 <= a <= 1) and (0 <= b <= 1)):
            return 0

    dist_possible = []
    dist_possible.append(dist_point_line(line1X1, line1Y1, line2X1, line2Y1, line2X2, line2Y2))
    dist_possible.append(dist_point_line(line1X2, line1Y2, line2X1, line2Y1, line2X2, line2Y2))
    dist_possible.append(dist_point_line(line2X1, line2Y1, line1X1, line1Y1, line1X2, line1Y2))
    dist_possible.append(dist_point_line(line2X2, line2Y2, line1X1, line1Y1, line1X2, line1Y2))
    return min(dist_possible)

def dist_point_line(pointX, pointY, lineX1, lineY1, lineX2, lineY2) :
    changeX = lineX2 - lineX1
    changeY = lineY2 - lineY1
    if (changeX == 0 and changeY == 0):
        dist = np.hypot(pointX - lineX1, pointY - lineY1)
        return dist

    mag = changeX * changeX + changeY * changeY
    changeSlope = ((pointX - lineX1) * changeX + (pointY - lineY1) * changeY) / mag

    if (changeSlope < 0):
        changeX = pointX - lineX1
        changeY = pointY - lineY1
    elif (changeSlope > 1):
        changeX = pointX - lineX2
        changeY = pointY - lineY2
    else:
        changeX = pointX - lineX1 - (changeSlope * changeX)
        changeY = pointY - lineY1 - (changeSlope * changeY)

    dist = np.hypot(changeX, changeY)
    return dist

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """

    buffer = granularity / np.sqrt(2)
    for wall in walls:
        wallX1 = wall[0]
        wallY1 = wall[1]
        wallX2 = wall[2]
        wallY2 = wall[3]
        alienX = alien.get_centroid()[0]
        alienY = alien.get_centroid()[1]
        alienRad = alien.get_width()
        if (alien.is_circle()): # circle alien case
            if ((dist_point_line(alienX, alienY, wallX1, wallY1, wallX2, wallY2) - alienRad) < buffer or np.isclose((dist_point_line(alienX, alienY, wallX1, wallY1, wallX2, wallY2) - alienRad), buffer)):
                return True
        else: # oblong alien case
            alienHeadX = alien.get_head_and_tail()[0][0]
            alienHeadY = alien.get_head_and_tail()[0][1]
            alienTailX = alien.get_head_and_tail()[1][0]
            alienTailY = alien.get_head_and_tail()[1][1]
            if ((dist_line_line(alienHeadX, alienHeadY, alienTailX, alienTailY, wallX1, wallY1, wallX2, wallY2) - alienRad) < buffer or np.isclose((dist_line_line(alienHeadX, alienHeadY, alienTailX, alienTailY, wallX1, wallY1, wallX2, wallY2) - alienRad), buffer)):
                return True

    return False

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """

    for goal in goals:
        goalX = goal[0]
        goalY = goal[1]
        goalPos = np.array((goalX, goalY))
        goalRad = goal[2]
        alienX = alien.get_centroid()[0]
        alienY = alien.get_centroid()[1]
        alienPos = np.array((alienX, alienY))
        alienRad = alien.get_width()
        if (alien.is_circle()): # circle alien case
            if (np.linalg.norm(alienPos-goalPos) < (alienRad + goalRad) or 
                np.isclose(np.linalg.norm(alienPos-goalPos), (alienRad + goalRad))):
                return True
        else: # oblong alien case
            alienHeadX = alien.get_head_and_tail()[0][0]
            alienHeadY = alien.get_head_and_tail()[0][1]
            alienHeadPos = np.array((alienHeadX, alienHeadY))
            alienTailX = alien.get_head_and_tail()[1][0]
            alienTailY = alien.get_head_and_tail()[1][1]
            alienTailPos = np.array((alienTailX, alienTailY))
            distHead = np.linalg.norm(alienHeadPos-goalPos)
            distTail = np.linalg.norm(alienTailPos-goalPos)
            minDistEnds = np.minimum(distHead, distTail)
            finalDist = minDistEnds
            if (alienHeadX == alienTailX): # vertical
                if (alienHeadY > alienTailY):
                    if (goalY >= alienTailY and goalY <= alienHeadY):
                        distPerp = abs(goalX - alienX)
                        finalDist = np.minimum(distPerp, minDistEnds)
                else:
                    if (goalY >= alienHeadY and goalY <= alienTailY):
                        distPerp = abs(goalX - alienX)
                        finalDist = np.minimum(distPerp, minDistEnds)
            if (alienHeadY == alienTailY): # horizontal
                if (alienHeadX > alienTailX):
                    if (goalX >= alienTailX and goalX <= alienHeadX):
                        distPerp = abs(goalY - alienY)
                        finalDist = np.minimum(distPerp, minDistEnds)
                else:
                    if (goalX > alienHeadX and goalX < alienTailX):
                        distPerp = abs(goalY - alienY)
                        finalDist = np.minimum(distPerp, minDistEnds)
            if (finalDist < (alienRad + goalRad) or np.isclose(finalDist, (alienRad + goalRad))):
                return True
    return False

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """

    width = window[0]
    height = window[1]
    buffer = granularity / np.sqrt(2)
    xLowerBound = buffer
    yLowerBound = buffer
    xUpperBound = width - buffer
    yUpperBound = height - buffer
    alienX = alien.get_centroid()[0]
    alienY = alien.get_centroid()[1]
    alienRad = alien.get_width()
    if (alien.is_circle()): # circle alien case
        if (alienX - alienRad < xLowerBound or np.isclose(alienX - alienRad, xLowerBound)):
            return False
        if (alienX + alienRad > xUpperBound or np.isclose(alienX + alienRad, xUpperBound)):
            return False
        if (alienY - alienRad < yLowerBound or np.isclose(alienY - alienRad, yLowerBound)):
            return False
        if (alienY + alienRad > yUpperBound or np.isclose(alienY + alienRad, yUpperBound)):
            return False
    else: # oblong alien case
        alienHeadX = alien.get_head_and_tail()[0][0]
        alienHeadY = alien.get_head_and_tail()[0][1]
        alienTailX = alien.get_head_and_tail()[1][0]
        alienTailY = alien.get_head_and_tail()[1][1]
        if (alienHeadX == alienTailX): #vertical alien
            if (alienX - alienRad < xLowerBound or np.isclose(alienX - alienRad, xLowerBound)):
                return False
            if (alienX + alienRad > xUpperBound or np.isclose(alienX + alienRad, xUpperBound)):
                return False
            if (alienHeadY > alienTailY):
                if (alienTailY - alienRad < yLowerBound or np.isclose(alienTailY - alienRad, yLowerBound)):
                    return False
                if (alienHeadY + alienRad > yUpperBound or np.isclose(alienHeadY + alienRad, yUpperBound)):
                    return False
            else:
                if (alienHeadY - alienRad < yLowerBound or np.isclose(alienHeadY - alienRad, yLowerBound)):
                    return False
                if (alienTailY + alienRad > yUpperBound or np.isclose(alienTailY + alienRad, yUpperBound)):
                    return False
        else: # horizontal alien
            if (alienHeadX > alienTailX):
                if (alienTailX - alienRad < xLowerBound or np.isclose(alienTailX - alienRad, xLowerBound)):
                    return False
                if (alienHeadX + alienRad > xUpperBound or np.isclose(alienHeadX + alienRad, xUpperBound)):
                    return False
            else:
                if (alienHeadX - alienRad < xLowerBound or np.isclose(alienHeadX - alienRad, xLowerBound)):
                    return False
                if (alienTailX + alienRad > xUpperBound or np.isclose(alienTailX + alienRad, xUpperBound)):
                    return False
            if (alienY - alienRad < yLowerBound or np.isclose(alienY - alienRad, yLowerBound)):
                return False
            if (alienY + alienRad > yUpperBound or np.isclose(alienY + alienRad, yUpperBound)):
                return False

    return True

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0) 
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")