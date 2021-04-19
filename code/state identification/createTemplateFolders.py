import os
import os
rootPath = r'C:\Users\jiali\Desktop\MapElementDetection\code\state identification\sample images\templates'

# us state name
validStateNames = ['Alabama','Arkansas','Arizona','California',
    'Colorado','Connecticut','Delaware','Florida','Georgia','Iowa','Idaho','Illinois',
    'Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesota','Missouri',
    'Mississippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey',
    'New Mexico','Nevada','New York','Ohio', 'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
    'South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming']

for stateName in validStateNames:
    if not os.path.exists(rootPath + '\\' + stateName):
        os.makedirs(rootPath + '\\' + stateName)