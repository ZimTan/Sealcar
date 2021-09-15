import os
import time
import socket
import select
import json
import logging
import base64

from threading import Thread

from io import BytesIO
import numpy as np
from PIL import Image

import autocar_controller.car.car_interface as car_interface

logger = logging.getLogger(logging.info("INFO:"))

import re

# DEBUG
import datetime

def replace_float_notation(string):
    """
    Replace unity float notation for languages like
    French or German that use comma instead of dot.
    This convert the json sent by Unity to a valid one.
    Ex: "test": 1,2, "key": 2 -> "test": 1.2, "key": 2

    :param string: (str) The incorrect json string
    :return: (str) Valid JSON string
    """
    regex_french_notation = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+),'
    regex_end = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+)}'

    for regex in [regex_french_notation, regex_end]:
        matches = re.finditer(regex, string, re.MULTILINE)

        for match in matches:
            num = match.group('num').replace(',', '.')
            string = string.replace(match.group('num'), num)
    return string

class SDClient:
    def __init__(self, host, port, queue, poll_socket_sleep_time=0.01):
        self.host = host
        self.port = port
        self.poll_socket_sleep_sec = poll_socket_sleep_time

        self.queue_msg_recv = queue

        self.msg = None
        self.th = None

        # the aborted flag will be set when we have detected a problem with the socket that we can't recover from.
        self.aborted = False
        self.connect()


    def connect(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # connecting to the server 
        logger.info("connecting to %s:%d " % (self.host, self.port))

        try:
            self.s.connect((self.host, self.port))
        except ConnectionRefusedError as e:
            raise( Exception("Could not connect to server."))

        # time.sleep(pause_on_create)
        self.do_process_msgs = True

        self.th = Thread(target=self.proc_msg, args=(self.s,))
        self.th.start()


    def send(self, m):
        self.msg = m

    def send_now(self, msg):
        logger.debug("send_now:" + msg)
        self.s.sendall(msg.encode("utf-8"))

    def on_msg_recv(self, j):
        logger.debug("got:" + j['msg_type'])

    def stop(self):
        # signal proc_msg loop to stop, then wait for thread to finish
        # close socket
        self.do_process_msgs = False
        if self.th is not None:
            self.th.join()
        if self.s is not None:
            self.s.close()

    def proc_msg(self, sock):
        '''
        This is the thread message loop to process messages.
        We will send any message that is queued via the self.msg variable
        when our socket is in a writable state. 
        And we will read any messages when it's in a readable state and then
        call self.on_msg_recv with the json object message.
        '''
        sock.setblocking(0)
        inputs = [ sock ]
        outputs = [ sock ]
        partial = []

        while self.do_process_msgs:
            # without this sleep, I was getting very consistent socket errors
            # on Windows. Perhaps we don't need this sleep on other platforms.
            time.sleep(self.poll_socket_sleep_sec)

            try:
                # test our socket for readable, writable states.
                readable, writable, exceptional = select.select(inputs, outputs, inputs)

                for s in readable:
                    # print("waiting to recv")
                    try:
                        data = s.recv(1024 * 256)
                    except ConnectionAbortedError:
                        logger.warn("socket connection aborted")
                        self.do_process_msgs = False
                        break
                    
                    # we don't technically need to convert from bytes to string
                    # for json.loads, but we do need a string in order to do
                    # the split by \n newline char. This seperates each json msg.
                    data = data.decode("utf-8")
                    msgs = data.split("\n")
                    #print(msgs)

                    for m in msgs:
                        if len(m) < 2:
                            continue
                        last_char = m[-1]
                        first_char = m[0]
                        # check first and last char for a valid json terminator
                        # if not, then add to our partial packets list and see
                        # if we get the rest of the packet on our next go around.                
                        if first_char == "{" and last_char == '}':
                            # Replace comma with dots for floats
                            # useful when using unity in a language different from English
                            m = replace_float_notation(m)
                            try:
                                j = json.loads(m)
                                j['time_treat'] = time.perf_counter()

                                self.queue_msg_recv.append(j)
                                if len(self.queue_msg_recv) > 3:
                                    self.queue_msg_recv.pop(0)

                            except Exception as e:
                                logger.error("Exception:" + str(e))
                                logger.error("json: " + m)
                        else:
                            partial.append(m)
                            # logger.info("partial packet:" + m)
                            if last_char == '}':
                                if partial[0][0] == "{":
                                    assembled_packet = "".join(partial)
                                    assembled_packet = replace_float_notation(assembled_packet)
                                    second_open = assembled_packet.find('{"msg', 1)
                                    if second_open != -1:
                                        # hmm what to do? We have a partial packet. Trimming just
                                        # the good part and discarding the rest.
                                        logger.warn("got partial packet:" + assembled_packet[:20])
                                        assembled_packet = assembled_packet[second_open:]

                                    try:
                                        j = json.loads(assembled_packet)
                                        j['time_treat'] = time.perf_counter()

                                        self.queue_msg_recv.append(j)

                                        if len(self.queue_msg_recv) > 3:
                                            self.queue_msg_recv.pop(0)

                                    except Exception as e:
                                        logger.error("Exception:" + str(e))
                                        logger.error("partial json: " + assembled_packet)
                                else:
                                    logger.error("failed packet.")
                                partial.clear()
                        
                for s in writable:
                    if self.msg != None:
                        logger.debug("sending " + self.msg)
                        s.sendall(self.msg.encode("utf-8"))
                        self.msg = None

                if len(exceptional) > 0:
                    logger.error("problems w sockets!")

            except Exception as e:
                print("Exception:", e)
                self.aborted = True
                self.on_msg_recv({"msg_type" : "aborted"})
                break

class SimClient(SDClient):
    """
      Handles messages from a single TCP client.
    """

    def __init__(self, address):
        self.queue_msg_recv = []
        super().__init__(*address, self.queue_msg_recv)

        while True:
            if len(self.queue_msg_recv) > 0:
                msg = self.queue_msg_recv.pop()
                break
            time.sleep(0.05)

        if msg['msg_type'] == 'car_loaded':
            print('client connected')

    def send_now(self, msg):
        # takes a dict input msg, converts to json string
        # and sends immediately. right now, no queue.
        json_msg = json.dumps(msg)
        super().send_now(json_msg)
    
    def send_control(self, control):
        msg = {'msg_type': 'control',
               'steering': str(control[0]),
               'throttle': str(control[1]),
               'brake': '0.0'}
        self.queue_message(msg)

    def queue_message(self, msg):
        # takes a dict input msg, converts to json string
        # and adds to a lossy queue that sends only the last msg
        json_msg = json.dumps(msg)
        self.send(json_msg)

    def on_msg_recv(self, jsonObj):
        # pass message on to handler
        self.msg_handler.on_recv_message(jsonObj)

    def get_msg(self):
        while True:
            if len(self.queue_msg_recv) > 0:
                msg = self.queue_msg_recv.pop()
                if msg['msg_type'] == 'telemetry':
                    msg['image'] = np.asarray(Image.open(BytesIO(base64.b64decode(msg['image']))))
                    return msg
            time.sleep(0.01)

    def is_connected(self):
        return not self.aborted

    def __del__(self):
        self.close()

    def close(self):
        # Called to close client connection
        self.stop()

class UnitySimulationClient(car_interface.CarInterface):
    def __init__(self, config):
        super().__init__(config)
        self.client = SimClient((self.config['host'], self.config['port']))
        self.prev_time = 0

    def get_data(self) -> dict:
        msg = self.client.get_msg()
        
        # Loop to prevent passed image to come back
        while self.prev_time > msg['time_treat']:
            self.prev_time = msg['time_treat']
            msg = self.client.get_msg()

        self.prev_time = msg['time_treat']
        return msg

    def send_action(self, action):
        self.client.send_control(action)

    def pause(self):
        """ Should return the boolean that stop the car """
        pass

    def stop(self):
        """ Should return the boolean that stop the car """
        self.client.stop()
