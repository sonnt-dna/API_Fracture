# References:
# https://learn.microsoft.com/en-us/azure/container-apps/github-actions

# ----------------------------------- NOTE - ONLY FOR ADMIN ----------------------------------------

# 1. Trong Subscription, cần cấp quyền cho Security Group có chứa Azure App
# 2. Trong Subscription, cần tạo trước 1 Resource Group có tên là "azc-container-app-rs-group"
# 3. Khi deploy lần đầu tiên sẽ gặp lỗi Register Microsoft.App và Microsoft.OperationalInsights
#    thì chỉ cần thoát ra và chạy lại file này là được, không cần xử lý gì thêm

# ---------------------------------- PARAMETERS -----------------------------------------------------

# String variables: Replace with your own
tenant_id = 'c5ec5abe-76c1-46cb-b3fe-c3b0071ffdb3'
# Subscription: huongltm@vpi.pvn.vn
subscription_id = '622682ed-510e-45fb-be7d-ebe44150c896'
resource_group_name = 'azc-container-app-rs-group'
container_app_environment_name = 'azc-environment'

# Tự động tạo tên của Azure Container App
# import datetime
# current_time = datetime.datetime.now()
# container_app_name = "azc-app-" + current_time.strftime("%Y-%m-%d-%H-%M")

# Delegated permission Power BI Service
# Để sử dụng được Azure App thì trong Subscription, cần cấp quyền cho Security Group có chứa Azure App
# với quyền "Owner" hoặc "Contributor" trong mục "Access control (IAM)"
# Security group: nhuduc_security_group_to_access_sharepoint
client_id = 'c3bf460c-3f80-48cb-ba7c-9597aa013552'
client_secret = 'MLV8Q~lxSThzcjq3rkSCvSZjvCBFy6Gjdzot8aIM' 

import subprocess
import re
import json
import sys
import platform

print('')
print('---------------------------------------- INPUT ------------------------------------------')
print('')

container_app_name = input("Nhập tên Container App theo quy tắc: Chữ thường, Không dấu, < 20 ký tự. Ví dụ: datamart-query :")
print('')
print('Install Azure CLI ...')
print('')
architecture = platform.machine()

# STEP 1: SETUP

# 1.1 - Install Azure CLI
if architecture == 'arm64':
    try:
        print("Updating Homebrew...")
        result = subprocess.run(['brew', 'update'], check=True, text=True, capture_output=True)
        result = subprocess.run(['brew', 'install', 'azure-cli'], check=True, text=True, capture_output=True)
        print("Successfully update Homebrew")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing Azure CLI: {str(e)}")
else:
    curl_command = "curl -v https://aka.ms/InstallAzureCLIDeb | sudo bash"
    try:
        subprocess.run(curl_command, shell=True, check=True)
        print("Successfully install Azure CLI")
    except subprocess.CalledProcessError as e:
        print("Lỗi khi thực thi lệnh curl:", e)

# 1.2 - Sign-in Azure CLI
command = f'az login --service-principal -u {client_id} -p {client_secret} --tenant {tenant_id}'
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode != 0:
    print(f'Error: {stderr.decode()}')
    raise SystemExit("Lỗi khi thực hiện lệnh: az login --service-principal -u {client_id} -p {client_secret} --tenant {tenant_id}")
else:
    print('Successfully Login as Service Principal!')
    # print(f'Successfully signin: {stdout.decode()}')
    if architecture == 'arm64':
        print('Waiting next step...')
    elif architecture == 'i386' or architecture == 'x86_64':
        print('')
        print('NOTE:')
        print('')
        print('Press ENTER to Continue...')

# 1.3 - Get list of container apps in resource group
command = f"az containerapp list --subscription {subscription_id} --resource-group {resource_group_name} --query [].name -o json"
result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
container_app_list_str = result.stdout
# print(type(container_app_list_str))
if container_app_list_str == None:
    print('Chưa có Container App nào được tạo')
else:
    print('Danh sách các Container App đã có: ')
    print(container_app_list_str)
    # Parse output to a list of names
    # container_app_list = json.loads(container_app_list_json)
    container_app_list = container_app_list_str.strip('[]').split(', ')
    # print(type(container_app_list))
    # 1.4 - Check if the input app name is in the list of names
    if container_app_name in container_app_list:
        print("Đã có Container App với tên tương tự, hãy đặt tên khác")
        sys.exit()
    else:
        print("Successfully check name of Container App in the Resrouce group!")
        print('Waiting next step ...')
        # print("Danh sách tên của container app trong resource group: ")
        # print('\n'.join(container_app_list))

    #--------------------------------- PROCESSING ------------------------------------------------

    # 1.5 - Install Azure Container App extension for CLI
    
    # Cấu hình để cài đặt extension tự động không dừng chờ nhắc
    try:
        print("Setting Azure CLI configuration...")
        result = subprocess.run(['az', 'config', 'set', 'extension.use_dynamic_install=yes_without_prompt'], check=True, text=True, capture_output=True)
        print("Azure CLI configuration set successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while setting Azure CLI configuration: {str(e)}")


    # Cài đặt extention Container App
    command = ['az', 'extension', 'add', '--name', 'containerapp', '--upgrade']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        # print(result.stdout)
        print("Successfully install Azure Container App extension for CLI")
        print('Waiting next step...')
    else:
        print(result.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh: az extension add --name containerapp --upgrade")

    # 1.6 - Register Microsoft.App
    command = ['az', 'provider', 'register', '--namespace', 'Microsoft.App', '--subscription', subscription_id]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        #print(result.stdout)
        print("Successfully Register Microsoft.App")
        print('Waiting next step...')
    else:
        print(result.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh: az provider register --namespace Microsoft.App")

    # 1.7 - Register Microsoft.OperationalInsights
    command = ['az', 'provider', 'register', '--namespace', 'Microsoft.OperationalInsights', '--subscription', subscription_id]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        #print(result.stdout)
        print("Successfully Register Microsoft.OperationalInsights")
        print('Waiting next step...')
    else:
        print(result.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh: az provider register --namespace Microsoft.OperationalInsights")


    # STEP 2+3:
    # Create Container Registry, Container App, Container App Environment and buil Docker Image
    # Azure CLI sẽ kiểm tra, resource nào đã có rồi (kiểm tra theo tên) thì sẽ không tạo nữa
    # resource nào chưa có thì sẽ tạo mới.
    # Riêng Container Registry bắt buộc phải để tên mặc định do Azure tự đặt, không thể tự đặt tên được
    # Muốn thiết lập "ingress" là "external" thì BẮT BUỘC phải thiết lập gía trị  cho "target-port" trước
    # 'Southeast Asia'      'West US 2'
    # Chú ý nếu bị lỗi có thể thử chuyển "Location" sang region khác để thử xem có chạy được không
    command = ['az', 'containerapp', 'up', 
                '--name', container_app_name, 
                '--source', '.', 
                '--ingress', 'external', 
                '--target-port', '3500',
                '--location', 'Southeast Asia',
                '--subscription', subscription_id, 
                '--resource-group', resource_group_name, 
                '--environment', container_app_environment_name]
                # '--service-principal-client-id', client_id,
                # '--service-principal-client-secret', client_secret,
                # '--service-principal-tenant-id', tenant_id]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        #print("result of container app up: " + result.stdout)
        print("Seccessfully Containerapp Up!")
        print('Waiting next step...')
        pattern = r"\b\w+\.azurecr\.io"
        match = re.search(pattern, result.stdout)
        if match:
            registry_name = match.group().replace(".azurecr.io", "")
            # print("registry name:--" + registry_name + "--")
    else:
        print(result.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh: az containeapp up")
        
    # STEP 4: Get Container Registry's Resrouce ID
    command = ['az', 'acr', 'show',
                '--name', registry_name,
                '--query', 'id',
                '--output', 'tsv',
                '--subscription', subscription_id, 
                '--resource-group', resource_group_name]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print('Successfully get Container Registry Resrouce ID')
        print('Waiting next step...')
        registry_resource_id = result.stdout
        # print("registry_resrouce_id:--" + registry_resource_id + "--")
    else:
        print(result.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh: az acr show --name <ACR_NAME> --query id --output tsv")

    # STEP 5: Enable managed identity for the container app
    command = ['az', 'containerapp', 'identity', 'assign',
                '--name', container_app_name,
                '--output', 'tsv',
                '--system-assigned',
                '--subscription', subscription_id, 
                '--resource-group', resource_group_name]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print('Successfully create identity')
        print('Waiting next step...')
        # print("stdout: " + result.stdout)
        managed_identity_principal_id = result.stdout.split()[0]
        print("Principal ID of the managed identity:--" + managed_identity_principal_id + "--")
    else:
        print(result.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh: az containerapp identity assign")

    # STEP 6: Assign the AcrPull role for the Azure Container Registry to the container app's managed identity
    # Với Service Principal cần thêm 2 parameter là '--assignee-object-id' và '--assignee-principal-type'
    # '--assignee-object-id', client_id,             '--assignee-principal-type', 'ServicePrincipal',             '--resource-group', resource_group_name]              '--debug',
    command = ['az', 'role', 'assignment', 'create',
                '--assignee', managed_identity_principal_id,
                '--role', 'AcrPull',        
                '--scope', registry_resource_id]
    try:
        print('Successfully assign role AcrPull to Identity')
        print('Waiting next step...')
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        #print("stdout: " + result.stdout)
    except subprocess.CalledProcessError as e:
        print('')
        #print(e.stderr)
        #raise SystemExit("Lỗi khi thực hiện lệnh: az role assignment create")

        
    # STEP 7: Pull images from the Azure Container Registry.
    command = ['az', 'containerapp', 'registry', 'set',
                '--name', container_app_name,
                '--server', registry_name + ".azurecr.io",
                '--identity', 'system',
                '--subscription', subscription_id, 
                '--resource-group', resource_group_name]
    try:
        print('Successfully Pull image from Registry to Container App')
        print('Waiting next step...')
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # print("stdout: " + result.stdout)
    except subprocess.CalledProcessError as e:
        print('')
        #print(e.stderr)
        #raise SystemExit("Lỗi khi thực hiện lệnh: az containerapp registry set")
    
    # STEP 8: Check Provisioning State
    command = ['az', 'containerapp', 'show',
            '--name', container_app_name,
            '--subscription', subscription_id, 
            '--resource-group', resource_group_name,
            '--query', 'properties.provisioningState']
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            provisioning_state = result.stdout.replace('"', '')
        else:
            print(result.stderr)
            raise SystemExit("Lỗi khi thực hiện lệnh kiểm tra Provision State")
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh Provisiont State")

    # STEP 9: Check Provisioning State
    command = ['az', 'containerapp', 'show',
            '--name', container_app_name,
            '--subscription', subscription_id, 
            '--resource-group', resource_group_name,
            '--query', 'properties.runningStatus']
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            running_state = result.stdout.replace('"', '')
        else:
            print(result.stderr)
            raise SystemExit("Lỗi khi thực hiện lệnh kiểm tra Running State")
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh Running State")
    
    # STEP 10: Show Endpoint of Container App
    command = ['az', 'containerapp', 'show',
            '--name', container_app_name,
            '--subscription', subscription_id, 
            '--resource-group', resource_group_name,
            '--query', 'properties.configuration.ingress.fqdn']
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.returncode == 0:
            endpoint = result.stdout.replace('"', '')
            print('')
            print('------------------------------------- RESULT -----------------------------------')
            print('')
            print('Endpoint of Azure Container App:')
            print('')
            print('https://' + endpoint)
            print('')
            print('Provision Status: ' + provisioning_state)
            print('Running Status: ' + running_state)
            print('-------------------------------------- END -------------------------------------')
        else:
            print(result.stderr)
            raise SystemExit("Lỗi khi thực hiện lệnh Show Endpoint")
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise SystemExit("Lỗi khi thực hiện lệnh Show Endpoint")
  