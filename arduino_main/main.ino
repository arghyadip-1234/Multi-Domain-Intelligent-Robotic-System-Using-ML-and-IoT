#include <Servo.h>

Servo servo[8];
int default_angle[8] = {75, 90, 90, 60, 75, 90, 90, 60};

void setup()
{
    Serial.begin(115200);

    servo[0].attach(5);
    servo[1].attach(6);
    servo[2].attach(7);
    servo[3].attach(8);
    servo[4].attach(1);
    servo[5].attach(2);
    servo[6].attach(3);
    servo[7].attach(4);

    for (size_t i = 0; i < 8; i++)
    {
        servo[i].write(default_angle[i]);
    }
}

byte angle[8];
byte pre_angle[8];
unsigned long t = millis();

void loop()
{
    if (Serial.available() >= 8)
    {
        Serial.readBytes(angle, 8);
        for (size_t i = 0; i < 8; i++)
        {
            if (angle[i] != pre_angle[i])
            {
                servo[i].write(angle[i]);
                pre_angle[i] = angle[i];
            }
        }
        t = millis();
    }

    if (millis() - t > 1000)
    {
        for (size_t i = 0; i < 8; i++)
        {
            servo[i].write(default_angle[i]);
            pre_angle[i] = default_angle[i];
        }
    }
}