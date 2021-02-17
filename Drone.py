# -*- coding: utf-8 -*-
class Drone:
    __name = ''
    __battery_time = 0  # minutes
    __speed = 0  # m/s
    __image_size = (0, 0)  # (width, height)
    __height = 0  # m

    def __init__(self, name, battery_time, speed, image_size, height, image_angle):
        self.__name = name
        self.__battery_time = battery_time
        self.__speed = speed
        self.__image_size = image_size
        self.__height = height
        self.__image_angle = image_angle

    def get_name(self):
        return self.__name

    def get_battery_time(self):
        return self.__battery_time

    def get_speed(self):
        return self.__speed

    def get_image_size(self):
        return self.__image_size

    def get_height(self):
        return self.__height

    def get_image_angle(self):
        return self.__image_angle
