import leap
import time
import keyboard


class MyListener(leap.Listener):
    def __init__(self):
        print("Initialized")
        
        super().__init__()
        self.last_print_time = 0.0

        
        self.capture_key = "s"
        self.dataset_file = open(r"C:\Development\leapc-python-bindings-main\Dataset\Handsigns.csv", "a")
        self.current_label = "L" # Preprocess

        
    def on_connection_event(self, event):
        print("Leap Service Connected")


    def on_connection_lost_event(self, event):
        print("Leap Service Disconnected")


    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print("Connected")
        print(f"Found device {info.serial}")

    
    def on_device_lost_event(self, event):
        print("Disconnected")

    
    def on_exit_event(self, event):
        print("Exited")


    def on_tracking_event(self, event):
        current_time = time.time()

        # Print every 0.2 seconds
        if current_time - self.last_print_time < 0.2:
            return

        self.last_print_time = current_time

        # Capture only if at least 1 hand and s pressed
        if not keyboard.is_pressed(self.capture_key):
            return
        
        if len(event.hands) < 1:
            return
        
        # Timestamp
        ts_us = event.info.timestamp

        # Sample
        output = []

        output.append(self.current_label)

        # Frame info
        output.append(str(event.tracking_frame_id))
        output.append(str(ts_us))
        output.append(str(ts_s))
        output.append(str(len(event.hands)))

        # Hand info (type, confidence, palm, fingers and bones)
        for hand in event.hands:

            hand_type = "left" if str(hand.type) == "HandType.Left" else "right" # Preprocess
            hand_confidence = hand.confidence
            normal = hand.palm.normal
            palm = hand.palm.position

            output.append(hand_type)
            output.append(str(hand_confidence))

            output.extend([
                str(palm.x), str(palm.y), str(palm.z),
                str(normal.x), str(normal.y), str(normal.z)
            ])

            # Fingers and Bones
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']

            for i, digit in enumerate(hand.digits):
                name = finger_names[i] if i < len(finger_names) else f"digit_{i}"

                # Fingertip positions 
                tip = digit.distal.next_joint
                output.extend([str(tip.x), str(tip.y), str(tip.z)])
                bones = [digit.metacarpal, digit.proximal, digit.intermediate, digit.distal]

                for bone_name, bone in zip(bone_names, bones):

                    pj = bone.prev_joint
                    nj = bone.next_joint

                    output.extend([
                        str(pj.x), str(pj.y), str(pj.z),
                        str(nj.x), str(nj.y), str(nj.z)
                    ])

                # Extended state
                is_extended = digit.is_extended
                output.append("EXTENDED" if is_extended else "NOT EXTENDED") # Preprocess

        row = ",".join(output)
        print(row)
        self.dataset_file.write(row + "\n")


def main():
    my_listener = MyListener()

    connection = leap.Connection()

    connection.add_listener(my_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while running:
            time.sleep(1)


if __name__ == "__main__":
    main()
