import psycopg2
import uuid
import random
from datetime import datetime, timedelta

# Configuration
conn = psycopg2.connect(
    dbname="udevs_4a17598bc0d14a18869d1b40d22ba7f3_p_postgres_svcs",
    user="udevs_4a17598bc0d14a18869d1b40d22ba7f3_p_postgres_svcs",
    password="7FZNexacHb",
    host="142.93.164.37",
    port="30032",

)
# from ucode_sdk import Config, new

# api = new(Config(app_id="P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL", base_url="https://api.client.u-code.io"))

cursor = conn.cursor()

human_ids = [
    'a709df2c-0c2d-4f24-aba0-28ba403681d7',
    '63bd7a4a-cf1a-4cc1-96c2-d538008054f7',
    '03e21f08-7136-49e6-a484-c479d39f92f0',
    # '517c510f-5a1c-420f-a630-b4eb2688ab69',
    '60ed7d3a-b24e-4e45-978c-26df840529b0'
]

i = 1
start_date = datetime.today().date() - timedelta(days=23)
end_date = datetime.today().date() + timedelta(days=22)

for single_date in (start_date + timedelta(n) for n in range((end_date - start_date).days)):
    for hour in range(24):
        for human_id in human_ids:
            i+=1
            print(i)
            guid = str(uuid.uuid4())
            created_at = updated_at = datetime.combine(single_date, datetime.min.time()) + timedelta(hours=hour)
            smile_count = random.randint(10, 70)
            normal_count = random.randint(1, 100)
            upset_count = random.randint(20, 100)
            smile_duration = round(random.uniform(0.1, 10.0), 2)
            normal_duration = round(random.uniform(0.1, 30.0), 2)
            upset_duration = round(random.uniform(0.1, 10.0), 2)
            
            # _, _, err = api.items("detection").create({
            #     "smile_count" : smile_count,
            #     "normal_count" : normal_count,
            #     "upset_count" : upset_count,
            #     "smile_duration" : smile_duration,
            #     "normal_duration" : normal_duration,
            #     "upset_duration" : upset_duration
            # }).exec()
            # if err:
            #     print(err)
                
            cursor.execute("""
                INSERT INTO detection (
                    guid, folder_id, created_at, updated_at, deleted_at, human_id, date, hour, person_type,
                    smile_count, normal_count, upset_count, normal_duration, upset_duration, smile_duration, company_id
                ) VALUES (
                    %s, NULL, %s, %s, NULL, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, NULL
                )
            """, (
                guid, created_at, updated_at, human_id, single_date, hour, ['employee'],
                smile_count, normal_count, upset_count,
                normal_duration, upset_duration, smile_duration
            ))

# Commit and close
conn.commit()
    
cursor.close()
conn.close()


