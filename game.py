import pygame
import time
import math
import random
import numpy as np
import pickle
import os

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Car Racing")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

car_size = (40, 20)
start_x, start_y = 325, 475
car_x, car_y = start_x, start_y
car_angle = 0
speed = 0
max_speed = 3
acceleration = 0.1
friction = 0.05
turn_speed = 3

car_surf = pygame.Surface(car_size, pygame.SRCALPHA)
pygame.draw.rect(car_surf, RED, (0, 0, *car_size))
pygame.draw.circle(car_surf, YELLOW, (car_size[0] - 5, 5), 3)
pygame.draw.circle(car_surf, YELLOW, (car_size[0] - 5, car_size[1] - 5), 3)

track_outer = [(150, 500), (650, 500), (750, 400), (750, 200), (650, 100), 
               (150, 100), (50, 200), (50, 400), (150, 500)]
track_inner = [(300, 450), (550, 450), (700, 325), (700, 250), (550, 175), 
               (250, 175), (100, 250), (100, 325), (250, 450)]

checkpoints = [
    (500, 475), (650, 435), (725, 350), (725, 240), 
    (640, 165), (500, 140), (300, 140), (160, 165),
    (75, 240), (75, 350), (150, 435), (300, 475)
]
checkpoint_radius = 30
checkpoint_attraction_radius = 120
checkpoint_index = 0

finish_line = (350, 475)
finish_x, finish_y = finish_line
finish_y_range = (450, 500)
finish_start = (finish_x, finish_y_range[0])
finish_end = (finish_x, finish_y_range[1])

timer_running = False
start_time = 0
lap_time = 0
last_lap_time = 0
best_lap_time = float('inf')
has_crossed_finish = False

font = pygame.font.Font(None, 36)

iteration_count = 0
alpha = 0.1
gamma = 0.9
epsilon = 0.3
min_epsilon = 0.01
epsilon_decay = 0.995
Q_table = {}

class AIController:
    def __init__(self):
        self.ray_count = 5
        self.ray_length = 150
        self.ray_angles = [-40, -20, 0, 20, 40]
        self.ray_hits = [self.ray_length] * self.ray_count
        self.rays = []
        self.steering = 0
        self.throttle = 0
        self.last_checkpoint = 0
        self.progress = 0
        self.best_run_states = []
        self.current_run_states = []
        self.last_checkpoint_time = 0

    def update_sensors(self, car_x, car_y, car_angle, track_mask_surface):
        self.rays = []
        self.ray_hits = []
        for angle in self.ray_angles:
            total_angle = car_angle + angle
            rad_angle = math.radians(total_angle)
            hit_distance = self.ray_length
            hit_x, hit_y = car_x, car_y

            for d in range(5, self.ray_length, 5):
                check_x = int(car_x + math.cos(rad_angle) * d)
                check_y = int(car_y - math.sin(rad_angle) * d)

                if 0 <= check_x < WIDTH and 0 <= check_y < HEIGHT:
                    if track_mask_surface.get_at((check_x, check_y)) == (0, 0, 0, 255):
                        hit_distance = d
                        hit_x = car_x + math.cos(rad_angle) * hit_distance
                        hit_y = car_y - math.sin(rad_angle) * hit_distance
                        break

            self.rays.append(((car_x, car_y), (hit_x, hit_y)))
            self.ray_hits.append(hit_distance)

    def get_sensor_data(self):
        return tuple(min(int(h/20), 7) for h in self.ray_hits)

    def make_decision(self, state):
        if state not in Q_table:
            Q_table[state] = [random.uniform(-0.5, 0.5) for _ in range(5)]
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 4)
        return np.argmax(Q_table[state])

    def apply_action(self, action):
        action_map = {
            0: (-0.5, 0.7), 1: (-0.2, 0.9), 2: (0, 1.0), 
            3: (0.2, 0.9), 4: (0.5, 0.7)
        }
        self.steering, self.throttle = action_map[action]

    def check_checkpoint(self, car_x, car_y):
        global checkpoint_index
        current_time = time.time()
        
        if current_time - self.last_checkpoint_time < 0.3:
            return False, 0
            
        cx, cy = checkpoints[checkpoint_index]
        distance = math.dist((car_x, car_y), (cx, cy))
        
        if distance < checkpoint_radius * 1.5:
            center_bonus = (1 - (distance / checkpoint_radius)) * 5
            checkpoint_index = (checkpoint_index + 1) % len(checkpoints)
            self.progress = checkpoint_index
            self.last_checkpoint_time = current_time
            print(f"Checkpoint {checkpoint_index+1}/{len(checkpoints)} reached (Distance: {distance:.1f})")
            return True, center_bonus
        return False, 0

    def get_checkpoint_attraction(self, car_x, car_y):
        """Calculate directional pull toward current checkpoint"""
        cx, cy = checkpoints[checkpoint_index]
        dx = cx - car_x
        dy = cy - car_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < checkpoint_attraction_radius:
            scale = 0.3 * (1 - distance/checkpoint_attraction_radius)
            return (dx/distance * scale, dy/distance * scale)
        return (0, 0)

    def get_state(self, car_x, car_y, car_angle):
        sensor_data = self.get_sensor_data()
        progress = min(self.progress, len(checkpoints)-1)
        angle = int(car_angle % 360 // 15)
        
        cx, cy = checkpoints[checkpoint_index]
        rel_x = cx - car_x
        rel_y = cy - car_y
        checkpoint_dir = int(math.degrees(math.atan2(-rel_y, rel_x)) % 360 // 30)
        
        return (progress, checkpoint_dir) + sensor_data + (angle,)

track_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
pygame.draw.polygon(track_surface, GRAY, track_outer)
pygame.draw.polygon(track_surface, WHITE, track_inner)

track_mask_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
track_mask_surface.fill((0, 0, 0, 0))
pygame.draw.lines(track_mask_surface, (0, 0, 0, 255), True, track_outer, 5)
pygame.draw.lines(track_mask_surface, (0, 0, 0, 255), True, track_inner, 5)

track_mask = pygame.mask.from_surface(track_mask_surface)
ai_controller = AIController()
human_control = False

def reset_car():
    global car_x, car_y, car_angle, speed, timer_running, lap_time, checkpoint_index
    if random.random() < 0.3 and len(ai_controller.best_run_states) > 10:
        for s in ai_controller.best_run_states[:10]:
            if s in Q_table:
                Q_table[s] = [q * 1.1 for q in Q_table[s]]
    
    car_x, car_y = start_x, start_y
    car_angle = 0
    speed = 0
    checkpoint_index = 0
    ai_controller.progress = 0
    ai_controller.current_run_states = []
    ai_controller.last_checkpoint_time = 0
    if timer_running:
        timer_running = False
        lap_time = 0

def save_progress():
    with open('q_learning_progress.pkl', 'wb') as f:
        pickle.dump({
            'Q_table': Q_table, 
            'iteration_count': iteration_count, 
            'epsilon': epsilon,
            'best_run_states': ai_controller.best_run_states
        }, f)

def load_progress():
    global Q_table, iteration_count, epsilon
    if os.path.exists('q_learning_progress.pkl'):
        with open('q_learning_progress.pkl', 'rb') as f:
            progress = pickle.load(f)
            Q_table = progress['Q_table']
            iteration_count = progress['iteration_count']
            epsilon = progress.get('epsilon', 0.3)
            if 'best_run_states' in progress:
                ai_controller.best_run_states = progress['best_run_states']
        print("Progress loaded!")
    else:
        print("No saved progress found.")

load_progress()

running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                human_control = not human_control
            elif event.key == pygame.K_r:
                reset_car()
            elif event.key == pygame.K_s:
                save_progress()

    ai_controller.update_sensors(car_x, car_y, car_angle, track_mask_surface)
    prev_x, prev_y = car_x, car_y
    
    if not human_control:
        state = ai_controller.get_state(car_x, car_y, car_angle)
        ai_controller.current_run_states.append(state)
        action = ai_controller.make_decision(state)
        ai_controller.apply_action(action)
        car_angle += ai_controller.steering * 10
        speed = ai_controller.throttle * max_speed

    dx = car_x - prev_x
    dy = car_y - prev_y
    forward_motion = math.cos(math.radians(car_angle)) * dx - math.sin(math.radians(car_angle)) * dy
    
    attract_x, attract_y = ai_controller.get_checkpoint_attraction(car_x, car_y)
    car_x += attract_x
    car_y += attract_y

    car_x += math.cos(math.radians(car_angle)) * speed
    car_y -= math.sin(math.radians(car_angle)) * speed

    if (prev_x < finish_x <= car_x or prev_x > finish_x >= car_x) and finish_y_range[0] <= car_y <= finish_y_range[1]:
        if not has_crossed_finish:
            if timer_running:
                last_lap_time = time.time() - start_time
                if last_lap_time < best_lap_time:
                    best_lap_time = last_lap_time
                    if len(ai_controller.current_run_states) > len(ai_controller.best_run_states):
                        ai_controller.best_run_states = ai_controller.current_run_states.copy()
            else:
                timer_running = True
            start_time = time.time()
            has_crossed_finish = True
    elif abs(car_x - finish_x) > 10 or not (finish_y_range[0] <= car_y <= finish_y_range[1]):
        has_crossed_finish = False

    if timer_running:
        lap_time = time.time() - start_time

    rotated_car = pygame.transform.rotate(car_surf, car_angle)
    car_rect = rotated_car.get_rect(center=(car_x, car_y))
    car_mask = pygame.mask.from_surface(rotated_car)

    reward = 0
    if track_mask.overlap(car_mask, (car_rect.x, car_rect.y)):
        reward = -5
        iteration_count += 1
        reset_car()
    else:
        checkpoint_hit, center_bonus = ai_controller.check_checkpoint(car_x, car_y)
        if checkpoint_hit:
            reward = 15 + (checkpoint_index * 2) + center_bonus  # Increased base reward
            if len(ai_controller.current_run_states) > len(ai_controller.best_run_states):
                ai_controller.best_run_states = ai_controller.current_run_states.copy()
        else:
            motion_toward_checkpoint = dx * attract_x + dy * attract_y
            
            track_center = ((track_outer[0][0] + track_inner[0][0])/2, 
                           (track_outer[0][1] + track_inner[0][1])/2)
            distance_to_center = math.dist((car_x, car_y), track_center)
            center_reward = max(0, 1 - distance_to_center/200)
            
            reward = max(0, forward_motion) * 0.2 + \
                    max(0, motion_toward_checkpoint) * 0.5 + \
                    center_reward * 0.1

    keys = pygame.key.get_pressed()
    if human_control:
        if keys[pygame.K_LEFT]:
            car_angle += turn_speed
        if keys[pygame.K_RIGHT]:
            car_angle -= turn_speed
        if keys[pygame.K_UP]:
            speed = min(speed + acceleration, max_speed)
        if keys[pygame.K_DOWN]:
            speed = max(speed - acceleration, -max_speed / 2)
    else:
        state = ai_controller.get_state(car_x, car_y, car_angle)
        ai_controller.current_run_states.append(state)
        action = ai_controller.make_decision(state)
        ai_controller.apply_action(action)
        car_angle += ai_controller.steering * 10
        speed = ai_controller.throttle * max_speed

    # Apply friction to slow down the car when no input is given
    if human_control:
        if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            speed *= (1 - friction)


    if not human_control:
        next_state = ai_controller.get_state(car_x, car_y, car_angle)
        max_next_q = np.max(Q_table.get(next_state, [0]*5))
        Q_table[state] = Q_table.get(state, [0]*5)
        action_value = Q_table[state][action]
        Q_table[state][action] = (1 - alpha) * action_value + alpha * (reward + gamma * max_next_q)

    if iteration_count % 50 == 0 and epsilon > min_epsilon:
        epsilon *= epsilon_decay

    screen.fill(WHITE)
    screen.blit(track_surface, (0, 0))
    pygame.draw.lines(screen, BLACK, True, track_outer, 5)
    pygame.draw.lines(screen, BLACK, True, track_inner, 5)
    pygame.draw.line(screen, BLACK, finish_start, finish_end, 5)

    for i, (cx, cy) in enumerate(checkpoints):
        color = GREEN if i == checkpoint_index else YELLOW
        #pygame.draw.circle(screen, (*color[:3], 50), (int(cx), int(cy)), checkpoint_attraction_radius)
        #pygame.draw.circle(screen, color, (int(cx), int(cy)), checkpoint_radius)
        checkpoint_text = font.render(str(i+1), True, BLUE)
        text_rect = checkpoint_text.get_rect(center=(cx, cy))
        screen.blit(checkpoint_text, text_rect)

    if attract_x != 0 or attract_y != 0:
        end_x = car_x + attract_x * 100
        end_y = car_y + attract_y * 100
        pygame.draw.line(screen, BLUE, (car_x, car_y), (end_x, end_y), 2)

    for ray in ai_controller.rays:
        pygame.draw.line(screen, GREEN, ray[0], ray[1], 1)

    screen.blit(rotated_car, car_rect.topleft)

    info_texts = [
        font.render(f"Time: {lap_time:.2f}s", True, BLACK),
        font.render(f"Last: {last_lap_time:.2f}s", True, BLACK),
        font.render(f"Best: {best_lap_time:.2f}s", True, BLACK),
        font.render("Control: " + ("Human" if human_control else "AI"), True, BLACK),
        font.render("SPACE: Toggle AI/Human, R: Reset, S: Save", True, BLACK),
        font.render(f"Iteration: {iteration_count}", True, BLACK),
        font.render(f"Exploration: {epsilon:.3f}", True, BLACK),
        font.render(f"Checkpoint: {checkpoint_index+1}/{len(checkpoints)}", True, BLACK)
    ]

    screen.blit(info_texts[0], (10, 40))
    screen.blit(info_texts[1], (WIDTH - info_texts[1].get_width() - 10, 40))
    screen.blit(info_texts[2], (WIDTH//2 - info_texts[2].get_width()//2, 10))
    screen.blit(info_texts[5], (WIDTH//2 - info_texts[5].get_width()//2, 30))
    screen.blit(info_texts[6], (WIDTH//2 - info_texts[6].get_width()//2, 50))
    screen.blit(info_texts[7], (WIDTH//2 - info_texts[7].get_width()//2, 70))
    screen.blit(info_texts[3], (10, HEIGHT - 60))
    screen.blit(info_texts[4], (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

    if iteration_count % 100 == 0:
        save_progress()

pygame.quit()