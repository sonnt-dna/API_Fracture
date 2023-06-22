<p align = "center" draggable=”false” ><img src="http://vpi.pvn.vn/wp-content/uploads/2020/07/VPI_logo.png" 
     width="100px"
     height="auto"/>
</p>
<h2 align="center" id="heading">:wave: Hướng dẫn sử dụng Template để build và deploy API lên Azure Container</h2>

### Introduction

Template này được sử dụng để xây dựng Project API và deploy lên Azure Container App, cụ thể:
- Tạo 1 AP Project mới từ Template Repository có sẵn trên Github Codespace
- Đóng gói API Project thành Docker Container trên Github Codespace
- Test thử API bằng Postman Desktop
- Deploy API Project lên Azure Container App trên Github Codespace
- Push source code từ Github Codespace lên Github Repository

<!--- Note: Add a horizental line in the page --->
#

### Yêu cầu đối với người sử dụng

Để có thể sử dụng template này, người dùng cần đáp ứng các yêu cầu sau:

- Có tài khoản Azure. Nếu chưa có thì [Đăng ký tại đây ](https://azure.microsoft.com/en-us/free/?WT.mc_id=A261C142F)
- Có kiến thức [cơ bản về Docker Container](https://viblo.asia/p/docker-la-gi-kien-thuc-co-ban-ve-docker-maGK7qeelj2)
- Có kiến thức [cơ bản về FastAPI](https://viblo.asia/p/huong-dan-co-ban-framework-fastapi-tu-a-z-phan-1-V3m5W0oyKO7)
- Có kiến thức [cơ bản về Postman](https://viblo.asia/p/huong-dan-su-dung-postman-cho-test-api-aWj53Lb1K6m)

#

### Các bước thực hiện để xây dựng dự án API Project và Deploy lên Azure Container App

#### Bước 1: Tạo một Repository mới từ Template
- Trong Repository hiện tại, click **Use this template**, chọn **Create a new repository**
- Chọn **Owner** cho Repostory mới
- Chọn loại Repository là **Private**
- Đặt tên cho Repository mới
- Click **Create repository from template** để tạo Repo mới từ template

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/5.png?raw=true)

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/6.png?raw=true)

#### Bước 2: Mở Repo vừa tạo bằng Github Codespace và build API theo nhu cầu
- Mở Repository vừa tạo
- Click **<> Code** ở góc trên bên phải, chọn **Codespaces**
- Click **Create codespaces on main** để tạo một Github Codespace mới từ Repository hiện tại
- Build API theo nhu cầu
- **Lưu ý quan trọng:** 
    + Tất cả các file source code mới tạo đều **CHỈ TẠO** trong folder **Service**
    + **Không chỉnh sửa** các file ngoài folder **Service** (giữ nguyên nội dung)

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/7.png?raw=true)

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/8.png?raw=true)

#### Bước 3: Cài đặt các tools trên Github Codespace để debug và run
- Cài đặt extension "Docker"
- Cài đặt extension "Azure"
- Cài đặt Azure CLI:
    + Mở Terminal mới
    + Cài đặt Azure CLI:
    ```shell
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    ```

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/9.png?raw=true)

#### Bước 4: Build Docker Image và chạy thử
Sau khi xây dựng xong API thì cần build Docker Image và chạy thử để kiểm tra
- Mở Terminal
- Chuyển tới folder **src**:
     ```shell
    cd src
     ```
- Build Docker Image:
    ```shell
    docker build -t <image_name> .
    ```
     Lưu ý thay *<image_name>* bằng tên image cụ thể (tên bằng tiếng Anh, không có khoảng trắng)
- Run Docker container từ Image vừa tạo:
     ```shell
    docker run -p 8017:3500 my_image uvicorn app.main:app --reload --host 0.0.0.0 --port 3500
     ```
- Sau khi run Container sẽ được hỏi có Public endpoin không => Chọn **Make Public**

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/11.png?raw=true)

- Trên Terminal, chọn **PORTS** và copy endpoint của API tại *Local Address*

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/2.png?raw=true)

- Mở Postman Desktop và paste Endpoint ở trên vào để test

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/10.png?raw=true)

#### Bước 5: Deploy Project lên Azure Container App
Sau khi chạy thử thành công ở **Bước 4** thì ta sẽ deploy Project lên Azure Container App, toàn bộ quá trình deploy sẽ được thực hiện bằng Azure CLI trên Terminal của Codespace:
- **Lưu ý**: Trong Codespace, mỗi lần tắt đi và mở lại thì các package đã cài đặt sẽ bị mất đi, do đó mỗi lần close Codespace và mở lại thì phải cài lại gói Azure CLI như hướng dẫn ở **Bước 3**

- Login vào tài khoản Azure:
    ```shell
    az login --use-device-code
    ```
    Trên Terminal sẽ hiện ra code, copy code đó và paste vào trang đăng nhập của Azure để xác thực tài khoản

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/12.png?raw=true)

- Upgrade để cập nhật phiên bản Azure CLI mới nhất
    ```shell
    az upgrade
    ```
- Cài đặt Azure Container App extension for CLI:
    ```shell
    az extension add --name containerapp --upgrade
    ```
- Đăng ký Microsoft.App trong Azure subscriptions của bạn
    ```shell
    az provider register --namespace Microsoft.App
    ```
- Đăng ký Microsoft.OperationalInsights trong Azure subscription
    ```shell
    az provider register --namespace Microsoft.OperationalInsights
    ```
- Trên Terminal của Codespace, chuyển sang thư mục "src" chứa source code của Project
    ```shell
    cd src
    ```
- **Tạo mới** Azure Container App và Deploy source code lên Container App
   ```shell
    az containerapp up \
    --name <YOUR_CONTAINER_APP_NAME> \
    --source . \
    --ingress external
    ```
    Trong phần Output trên Terminal, copy lại tên của Azure Container Registry

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/15.png?raw=true)

    Lệnh **az containerapp up** sẽ thực hiện các công việc sau:
    + Tự động tạo mới (và đặt tên) 5 loại resrouce trên Azure Portal gồm: Azure Resrouce Group, Azure Container Environment, Azure Container Registry, Azure Container App, Log Analytics workspace.
    
        + Trong trường hợp **_chỉ tạo mới Azure Container App_** dựa trên các Resources **_đã có_** (Azure Container Environment,...) thì thực hiện theo [hướng dẫn về sử dụng lệnh az containerapp up](https://learn.microsoft.com/en-us/cli/azure/containerapp?view=azure-cli-latest#az-containerapp-up)

    + Build Docker Image từ source code trong thư mục hiện tại của Project

    + Deploy Docker Image vừa tạo lên Azure Container App

- Kiểm tra trên Azure Portal xem các Resource đã được tạo thành công chưa, ta có thể đổi tên các resources cho phù hợp

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/14.png?raw=true)

- Lấy **Resrouce ID** của Azure Container Registry
   ```shell
    az acr show 
    --name <YOUR_AZURE_CONTAINER_REGISTRY_NAME> 
    --query id 
    --output tsv 
    --resource-group <YOUR_RESOURCE_GROUP_NAME>
    ```
    Thay thế các giá trị _<YOUR_AZURE_CONTAINER_REGISTRY_NAME>_ và _<YOUR_RESOURCE_GROUP_NAME>_ bằng tên các resource mà bạn đã tạo

    Trong Output của Terminal, copy lại Resource ID của Container Registry

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/16.png?raw=true)

- Kích hoạt tính năng **Quản lý danh tính tự động** _(Managed Identity)_ cho Azure Container App
   ```shell
    az containerapp identity assign \
    --name <YOUR_CONTAINER_APP_NAME> \
    --resource-group <YOUR_RESOURCE_GROUP_NAME> \
    --system-assigned \
    --output tsv
    ```
    Thay thế các giá trị _<YOUR_CONTAINER_APP_NAME>_ và _<YOUR_RESOURCE_GROUP_NAME>_ bằng tên các resource mà bạn đã tạo

    Trong Output của Terminal, copy lại Principal ID của Managed Identity (cột đầu tiên của Outpu)

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/22.png?raw=true)

- Gán Role **AcrPull** cho Azure Container App Registry
    ```shell
    az role assignment create \
    --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
    --role AcrPull \
    --scope <ACR_RESOURCE_ID>
    ```
    Thay thế _<MANAGED_IDENTITY_PRINCIPAL_ID>_ bằng Principle ID của Managed Identity và _<ACR_RESOURCE_ID>_ bằng Resource ID của Container Registry đã tạo ở 2 bước trước

- Cấu hình để cho phép Container App pull Docker Image từ Container Registry
    ```shell
    az containerapp registry set \
    --name my-container-app \
    --resource-group my-container-app-rg \
    --server <YOUR_AZURE_CONTAINER_REGISTRY_NAME>.azurecr.io \
    --identity system
    ```
    Thay thế _<YOUR_AZURE_CONTAINER_REGISTRY_NAME>_ bằng tên của resource mà bạn đã tạo

- Kiểm tra **kết quả deploy lên Azure Portal** bằng cách:
    + Mở trình duyệt, truy cập Azure Portal
    + Trong Azure Portal, mở Resource Group đã tạo (có tên đã được đặt ở biến môi trường RESOURCE_GROUP ở Bước 5)
    + Kiểm tra xem 4 resource sau đã được tạo thành công chưa:
        * Containger Registry
        * Container Apps Environment
        * Container App
        * Log Analytics Workspace

            ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/13.png?raw=true)

- Kiểm tra xem trong Container App **đã xác thực tài khoản Github** chưa:
    * Trên Azure Portal, mở Container App vừa tạo
    * Trên Sidebar bên trái, chọn **Continuous Deployment**
    * Trong mục **Github Settings**, xem đã đăng nhập tài khoản Github chưa
    * Nếu chưa đăng nhập thì **Signin**

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/1.png?raw=true)

- Kiểm tra xem Endpoint **đã được expose ra bên ngoài** (cho phép các app khác từ bên ngoài gọi API) hay chưa
    * Trên Azure Portal, mở Container App vừa tạo
    * Trên Sidebar bên trái, chọn **Ingress**
    * Enble các mục như hình vẽ để cho phép gọi API từ bên ngoài Azure

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/20.png?raw=true)

- Sau khi deploy xong, copy Endpoint của API (trên Azure Container App) và chuyển sang Postman để test

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/21.png?raw=true)

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/3.png?raw=true)


#### Bước 6: Tạo Github Action Workflow mới bằng cách cấu hình file .yaml

- Tạo Credential cho Github Workflow để đăng ký với Azure
    ```shell
    az ad sp create-for-rbac \
    --name <YOUR_APP_CREDENTIAL_NAME> \
    --role contributor \
    --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<YOUR_RESOURCE_GROUP_NAME> \
    --sdk-auth \
    --output json
    ```
    Thay thế _<YOUR_APP_CREDENTIAL_NAME>, <SUBSCRIPTION_ID>, <YOUR_RESOURCE_GROUP_NAME>_ bằng tên các resource của bạn

    **Lưu ý:** Nếu bị lỗi "Phát hiện một đối tượng đã tồn tại có ID là ..." hoặc  "Không đủ quyền..." khi tạo Credential thì nguyên nhân là do đã tồn tại Credential với tên tương tự. Khi đó phải đổi tên của _<YOUR_APP_CREDENTIAL_NAME>_

    Trong phần Output trên Terminal, copy chuỗi JSON chứa Credential

    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/17.png?raw=true)

- Add Credential vừa tạo vào Github Secret để tự động đăng nhập và Deploy mỗi khi source code thay đổi
    + Đăng nhập Github
    + Mở Repository của bạn
    + Chọn Setting > Secret and variables > Add Repository Secret
    + Đặt tên cho Secret là **AZURE_CREDENTIALS**
    + Paste chuỗi JSON Credential vừa tạo ở bước trước vào mục **Value**

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/18.png?raw=true)

- Cấu hình Github Action Workflow bằng cách update file **.yaml** trong folder **_.github/workflows/build-and-push.yaml_**:
    + Thay thế tên của các Resources trong ảnh sau bằng tên Resources của bạn

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/19.png?raw=true)

- Enable Workflow trong Github Action:

    Khi tạo một Github Action từ file .yaml trong folder **_/.github/workflows_** của Project thì mặc định Github sẽ disable tính năng Workflow nên ta cần enable tính năng này bằng cách:
    + Đăng nhập Github và truy cập Repository chứa Project
    + Chọn **Action**
    + Trong Page thông báo, chọn **Enable Workflow**

- Kiểm tra xem Github Action Workflow có hoạt động không bằng cách:
    + Tạo một thay đổi nhỏ trong source code
    + Mở Github Action xem thông báo Workflow có hoạt động không

        ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/23.png?raw=true)
    
    + Test lại bằng Postman xem đã cập nhật lên Endpoint chưa

#### Bước 7: Push source code từ Github Codespace sang Github Repository
  Khi build Project trên Codespace thì source code không được tự động push sang Github Repository nên ta phải thực hiện thủ công
- Trên Codespace Terminal, chuyển tới thư mục gốc của Project
    ```shell
    cd <root_folder_of_project>
    ```
- Kiểm tra trạng thái của các tệp đã thay đổi
    ```shell
    git status
    ```
- Đưa các tệp có thay đổi vào staging area
    + *Add tất cả các file có thay đổi*
    ```shell
    git add .
    ```
    + *Hoặc chỉ add 01 file cụ thể*
    ```shell
    git add <tên file>
    ```
- Commit các thay đổi:
    ```shell
    git commit -m <message commit>
    ```
- Update lên Github repo
    ```shell
    git push
    ```
    ![Xem ảnh](https://github.com/Vietnam-Petroleum-Institute/template-project-api-for-azure-container-app/blob/main/instruction-images/4.png?raw=true)




#### Tài liệu tham khảo
Tài liệu này được xây dựng trên cơ sở tham khảo [hướng dẫn của Microsoft tại đây](https://learn.microsoft.com/en-us/azure/container-apps/github-actions) và có cập nhật bổ sung một số nội dung khác để đảm bảo hướng dẫn đầy đủ các bước từ bắt đầu đến kết thúc
