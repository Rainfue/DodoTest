# %% Импортирование библиотек
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import torch

# Проверка девайса
print(torch.cuda.is_available())

# %%
# Аргументы для запуска
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='Путь к видео')                        # Путь до файла с видео
parser.add_argument('--conf', type=float, required=False, default=0.5, help='Уверенность модели')   # Уверенность детекции модели
args = parser.parse_args()

# %% Основной класс для мониторинга состояний

class TableMonitor:
    # Ф-я для инициализации класса
    def __init__(
            self, 
            roi: tuple,
            report_path: str = 'report.txt'
    ):
        self.roi = roi
        self.current_state = "empty"    # empty, taked, apearence
        self.time_state_changed = None
        self.events = pd.DataFrame(
            columns=['table_state', 'start_time', 'end_time', 'state_duration']
        )
        self.pending_state = None
        self.pending_frames = 0
        self.stability_frames = 10
        self.report_path = report_path
        # Очищаем логи перед стартом
        open(report_path, 'w').close()

    # Ф-я для обновления состояния
    def update(self, persons_bboxes, timestamp):
        # Сохраняем первый timestamp
        if self.time_state_changed is None:
            self.time_state_changed = timestamp

        # Вычисляем новое состояние
        is_occupied = any(self.is_overlapping(bbox) for bbox in persons_bboxes)
        new_state = 'taked' if is_occupied else 'empty'
        
        if new_state != self.current_state:
                # Если человек подошел
                if new_state == 'taked' and self.current_state == 'empty':
                    new_state = 'arrived'
                    self.update_state(new_state, timestamp)
                # В остальных случаях
                else:
                    self.update_state(new_state, timestamp)

    # Ф-я для определения вхождения в зону интереса
    def is_overlapping(self, person_bbox) -> bool:
        rx1, ry1, rx2, ry2 = self.roi
        px1, py1, px2, py2 = person_bbox
        no_overlap = (
            rx2 < px1 or    # ROI левее
            rx1 > px2 or    # ROI правее
            ry2 < py1 or    # ROI выше
            ry1 > py2       # ROI ниже
        )
        # True если пересекает
        return not no_overlap
    
    # Ф-я для обновления событий
    def update_state(self, new_state, timestamp):
        # Если новое состояние отличается от ожидаемого — сбрасываем счётчик
        if new_state != self.pending_state:
            self.pending_state = new_state
            self.pending_frames = 0
            return

        self.pending_frames += 1

        # Только когда состояние продержалось N кадров подряд — фиксируем
        if self.pending_frames >= self.stability_frames:
            state_duration = timestamp - self.time_state_changed
            event = {
                'table_state': self.current_state,
                'start_time': self.time_state_changed,
                'end_time': timestamp,
                'state_duration': state_duration,
            }
            self.events.loc[len(self.events)] = event

            # Меняем состояние и сбрасываем счётчик
            self.current_state = new_state
            self.time_state_changed = timestamp
            self.pending_state = None
            self.pending_frames = 0

            self.update_logs(f'[logging] NEW STATE -- {self.current_state}')
    
    # Ф-я для логирования
    def update_logs(self, log_content: str, timestamp: bool = True):
        if timestamp:
            log_content = log_content + f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        with open(self.report_path, 'a') as f:
            f.write('\n'+log_content)


# %% # Выбор зоны интереса - ROI

cap = cv2.VideoCapture(args.video)
ret, first_frame = cap.read()

roi = cv2.selectROI('Select table ROI', first_frame, fromCenter=False)
cv2.destroyAllWindows()

# Преобразовываем получивашиеся x, y, w, h в координаты (точки)
roi = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
# Возвращаем видео к началу 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# %% Конфигурация

monitor = TableMonitor(roi)                     # Класс для мониторинга 
model = YOLO('yolo11m.pt', task='detect')       # Модель детекции
model.to('cuda')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h)) # Объект для записи видео

# Логирование
monitor.update_logs(f'[start] Model ver = {model.ckpt_path}')
monitor.update_logs('=='*10, timestamp=False)

# %% Запуск pipeline детекции + отслеживания столика
# Начинаем детекцию
while cap.isOpened():
    # Список для хранения координат найденных людей
    persons_bboxes = []
    ret, frame = cap.read()
    if not ret:
        break

    # Прогоняем фрэйм через модель
    res = model(frame)
    boxes = res[0].boxes

    color = (0, 0, 255) if monitor.current_state == 'taked' else (0, 255, 0)
    # Отрисовываем зону интереса
    overlay = frame.copy()

    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), color, -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)


    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), color, 2)
    # Добавляем подпись
    cv2.putText(frame, monitor.current_state, (roi[0], roi[1]-10),
        cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2
    )
    

    # Проходимся по все найденным боксам
    for box in boxes:
         if box.cls == 0 and box.conf > args.conf:
            coords = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            persons_bboxes.append(coords)

    # получаем временнУю метку
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    monitor.update(persons_bboxes, timestamp)
    cv2.imshow('table monitoring', frame)

    # Сохраняем кадр
    writer.write(frame)

    # При нажатии на q - процесс остановится
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы, удаляем окна
cap.release()
writer.release()
cv2.destroyAllWindows()

# %% Аналитика результата
# Выводим получившийся датафрейм
monitor.update_logs('=='*10, timestamp=False)
monitor.update_logs(monitor.events.head().to_string(), timestamp=False)
print(monitor.events.head())

# Удаляем все записи, в которых длительность меньше 10 cекунд (нерелевантные)
events = monitor.events[monitor.events['state_duration'] > 10]

# %%
# Среднее время пока стол пустой 
avg_wait_time = events[events['table_state']=='empty']['state_duration'].mean()
# Среднее время пока стол занят
avg_take_time = events[events['table_state']=='taked']['state_duration'].mean()

# Логирование
monitor.update_logs(
    f'Mean "empty" time: {avg_wait_time:.2f} сек ({avg_wait_time/60:.2f} мин)',
    timestamp=False
)
monitor.update_logs(
    f'Mean "taked" time: {avg_take_time:.2f} сек ({avg_take_time/60:.2f} мин)',
    timestamp=False
)


# %%
# Визуализация изменения состояния
fig, ax = plt.subplots(figsize=(14, 4))

colors = {'empty': 'green', 'taked': 'red', 'arrived': 'orange'}

for _, row in monitor.events.iterrows():
    ax.barh(0, row['state_duration'], left=row['start_time'],
            color=colors[row['table_state']], edgecolor='white', linewidth=0.5)

ax.set_xlabel('Время (сек)')
ax.set_title('Состояния стола по времени')
ax.set_yticks([])
plt.tight_layout()
plt.savefig('timeline.png')
plt.show()
