"""
 Manage remote access with authentification to either OneDrive account or SSH server
 It allows to upload/send file to a remote server.
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: March 2023
 Last updated: Adrien Dorise - March 2023



"""


import requests
from azure.identity import InteractiveBrowserCredential
import json
import driveConfig as odv
from paramiko import SSHClient, AutoAddPolicy

class OneDriveAccess():
    '''
    Class implementing acces to oneDrive storage API
    Solution is a mix of:
    https://pypi.org/project/msgraph-core/ for authentification with MS graph 
    https://github.com/pranabdas/Access-OneDrive-via-Microsoft-Graph-Python for OneDrive management
    '''
    def __init__(self, client_id, tenant_id):
        self.token, self.URL, self.HEADERS = self.authentificate(client_id=client_id, tenant_id=tenant_id)


    def authentificate(self, client_id, tenant_id):
        """Authentificate to MS Graph using Miscrosoft account. Authentification is performed in the browser.
        Authentification parameters have to be retrieved in MS Azure.

        Args:
            client_id (string): Application ID
            tenant_id (string): Organisation ID

        Returns:
            string: Authentification token given for the current session. To be used for further access to MS Graph
            URL: OneDrive root URL
            dict: HEADERS used when connecting to MS Graph
        """
        browser_credential = InteractiveBrowserCredential(client_id=client_id, tenant_id=tenant_id)
        token = browser_credential.get_token('Files.Read.All', 'Files.ReadWrite').token
        URL = 'https://graph.microsoft.com/v1.0/'
        HEADERS = {'Authorization': 'Bearer ' + token}
        response = requests.get(URL + 'me/drive/', headers = HEADERS)
        
        if (response.status_code == 200):
            response = json.loads(response.text)
            print('Connected to the OneDrive of', response['owner']['user']['displayName']+' (',response['driveType']+' ).', \
                '\nConnection valid for one hour. Reauthenticate if required.')
        elif (response.status_code == 401):
            response = json.loads(response.text)
            print('API Error! : ', response['error']['code'],\
                '\nSee response for more details.')
        else:
            response = json.loads(response.text)
            print('Unknown error! See response for more details.')
        return token, URL, HEADERS


    def driveStorageInfo(self):
        """Print storage information
        """
        response = json.loads(requests.get(self.URL + 'me/drive/', headers = self.HEADERS).text)
        used = round(response['quota']['used']/(1024*1024*1024), 2)
        total = round(response['quota']['total']/(1024*1024*1024), 2)
        print('Using', used, 'GB (', round(used*100/total, 2),'%) of total', total, 'GB.')


    def listFiles(self):
        """List files accessible in the root folder in the user's OneDrive
        """
        items = json.loads(requests.get(self.URL + 'me/drive/root/children', headers=self.HEADERS).text)
        items = items['value']
        for entries in range(len(items)):
            print(items[entries]['name'], '| item-id >', items[entries]['id']) 
            
    def getFile(self, drivePath, localPath):
        """Load a file from OneDrive and save it in a specified local folder

        Args:
            drivePath (string): File path in the user's OneDrive
            localPath (string): Path to store the uploaded file
        """
        print(f"Load drive {drivePath} to local {localPath} in progress...")
        url = 'me/drive/root:/' + drivePath + ':/content'
        url = self.URL + url
        data = requests.get(url, headers=self.HEADERS)
        with open(localPath, 'wb') as file:
            file.write(data.content)
        print("Loading done!")

            
    def uploadFile(self, localPath, drivePath):
        """Upload a file from local directory to user's OneDrive

        Args:
            localPath (string): Path to local file to upload
            drivePath (string): OneDrive path to store file
        """
        
        print(f"Upload local {localPath} to drive {drivePath} in progres...")
        url = self.URL + 'me/drive/root:/' + drivePath + ':/content'
        content = open(localPath, 'rb')
        response = json.loads(requests.put(url, headers=self.HEADERS, data = content).text)
        #print("Upload status: " + response['@odata.context'])
        print("Upload done!")

class SSHaccess():
    """
    Class implementing access to a SSH server
    """
    def __init__(self,name,user,mdp):
        self.client = self.authentificate(name,user, mdp)

        
    def authentificate(self, name, user=None, mdp=None):
        """Authentification process to a remote server using SSH
        Don't forget to close the connection when you are done by calling close() func!
        Args:
            name (string): Server name. Can also be IP address
            user (string, optional): User account connected with SSH. Defaults to None.
            mdp (string, optional): Password used for connection. Defaults to None.

        Returns:
            SSH client: Connection object
        """
        client = SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())
        client.connect(hostname = name, port=22, username=user, password=mdp)
        return client
    
    def close(self):
        """Close the connection with SSH server
        Don't forget to use it when you are done!
        """
        self.client.close()
    
    def cmd(self, str="df"):
        """Send a command to the server

        Args:
            str (str, optional): Command to send. Defaults to "df".
        """
        _stdin, _stdout,_stderr = self.client.exec_command(str)
        _stderr = _stderr.read().decode()
        print("cmd: " + str)
        if(len(_stderr) == 0): #No error
            print("answer: " + _stdout.read().decode())
        else: #Error
            print("answer: " + _stderr)
        
    def sudoCmd(self, mdp, str="sudo df",):
        """Command using super user authorisation

        Args:
            mdp (string): Super user password
            str (str, optional): Command to send. Defaults to "sudo df".
        """
        _stdin, _stdout,_stderr = self.client.exec_command(str)
        _stdin.write(mdp)
        _stderr = _stderr.read().decode()
        print("cmd: " + str)
        if(len(_stderr) == 0): #No error
            print("answer: " + _stdout.read().decode())
        else: #Error
            print("answer: " + _stderr)

    def getFile(self, remotePath, localPath):
        """Dowload a file from server to local folder

        Args:
            remotePath (string): file location on the server
            localPath (string): local save location
        """
        print(f"Load drive {remotePath} to local {localPath} in progress...")
        ftp_access = self.client.open_sftp()
        ftp_access.get(remotePath,localPath)
        ftp_access.close()
        print("Loading done!")
        
        
    def uploadFile(self, localPath, remotePath):
        """Upload a file from local folder to remote server

        Args:
            localPath (string): local file location
            remotePath (string): remote save location
        """
        print(f"Upload local {localPath} to drive {remotePath} in progres...")
        ftp_access = self.client.open_sftp()
        ftp_access.put(localPath, remotePath)
        ftp_access.close()
        print("Upload done!")        

if __name__ == "__main__":

    '''
    #OneDrive access
    #Connection to drive
    oneDrive = OneDriveAccess(odv.client_id,odv.tenant_id)
    
    #API test
    oneDrive.listFiles()
    oneDrive.driveStorageInfo()
    
    #csv test
    oneDrive.uploadFile("data/mouse Tue Feb 21 15:45:41 2023.csv", "TEST/Debug.csv")
    oneDrive.getFile("TEST/Debug.csv", "data/Debug.csv")
    
    #Video test
    #oneDrive.uploadFile("data/user Tue Feb 21 15:45:41 2023.avi", "TEST/DebugVideo.avi")
    #oneDrive.loadFile("TEST/DebugVideo.avi", "data/DebugVideo.avi")
    '''
    
    #SSH access
    #Open access
    ssh = SSHaccess(odv.ssh_host, odv.ssh_user,odv.ssh_password)
    ssh.cmd("ls Recordings/")
    '''
    #csv test
    ssh.uploadFile("data/mouse Tue Feb 21 15:45:41 2023.csv", "SSH_tests/Debug.csv")
    ssh.getFile("SSH_tests/Debug.csv", "data/Debug2.csv")
    ssh.cmd("ls SSH_tests/")

    #video test
    ssh.uploadFile("data/user Tue Feb 21 15:45:41 2023.avi", "SSH_tests/DebugVideo.avi")
    ssh.getFile("SSH_tests/DebugVideo.avi", "data/DebugVideo.avi")
    ssh.cmd("ls SSH_tests/")
    ''' 
    #Close connection
    ssh.close()