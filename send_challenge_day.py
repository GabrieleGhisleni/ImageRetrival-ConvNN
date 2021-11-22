import os, json, requests, time
from pprint import  pprint

def submit(results, url):
    i, send_again, check = 1,[], set()
    print(f"We have {len(results)} solution to send")
    for result in results:
        with open(result, "r") as file:
            tmp_result = json.load(file)
        res = json.dumps(tmp_result)
        response = requests.post(url, res)
        try:
            our_result = json.loads(response.text)
            name=result.split('-last')[1].split('.json')[0]
            print(f"Model n {i} --> {name} ---> {our_result['results']}")
            check.add((name, tuple(our_result['results'].items())))
            i += 1
        except Exception:
            print(f"Error --> Too many request in error : {'Too Many Requests' in response.text}")
            send_again.append(result)
        time.sleep(15)
    print(f"To send again ---> {len(send_again)}\n")
    while send_again != []:
        print(f"Have to do again --->{len(send_again)}<--- post because of errors")
        for result in send_again:
            with open(result, "r") as file:
                tmp_result = json.load(file)
            res = json.dumps(tmp_result)
            response = requests.post(url, res)
            try:
                our_result = json.loads(response.text)
                name = result.split('-last')[1].split('.json')[0]
                print(f"Model n {i} --> {name} ---> {our_result['results']}")
                i += 1
                check.add((name, tuple(our_result['results'].items())))
                send_again.remove(result)
            except Exception:
                print(f"Error --> Too many request in error : {'Too Many Requests' in response.text}")
            time.sleep(15)
    pprint("\n\n{}".format(check))

url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"
#url = "http://kamino.disi.unitn.it:3001/results/"
result = ["Results/" +  i for i in os.listdir("Results")]

# "name of the app GlobalApp" #
submit(result, url)
