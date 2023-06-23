# =============================================================================

import subprocess
from git import Repo, Actor
import os
from github import Github

def clone_from_github(user_name, user_email, repo_dir, branch_name, github_token, github_url):
    '''
    Clone file from GitHub
    :param user_name: GitHub username
    :param user_email: GitHub email
    :param repo_dir: Path to the repo
    :param branch_name: Branch name
    :param github_token: GitHub token
    :param github_url: GitHub url
    :return: None

    Example:
    clone_from_github('nguyenvana', 'nguyenvana@gmail.com', 'C:\\Users\\nguyenvana\\Documents\\GitHub\\test', ['test.txt'], 'Test', 'main', 'ghp_1234567890', 'https://github.com/nguyenvana.vn/test.git')
    '''
    os.system(f'git config --global user.email {user_email}')
    os.system(f'git config --global user.name {user_name}')

    # Clone repo
    if not os.path.exists(repo_dir):
        Repo.clone_from(f'https://token:{github_token}@{github_url[8:]}', repo_dir)
        print(f"Đã clone repo:{repo_dir} trên GitHub về Lakehouse thành công!")
    else:
        print(f"Đã tồn tại repo:{repo_dir} trên Lakehouse!")

# def clone_from_github(user_name, user_email, repo_dir, branch_name, github_token, github_url):

#     from git import Repo, Git
#     # Tạo repo mới
#     repo = Repo.init(repo_dir)
    
#     # Sử dụng config_writer để cài đặt thông tin user.name và user.email
#     with repo.config_writer() as git_config:
#         git_config.set_value('user', 'name', user_name)
#         git_config.set_value('user', 'email', user_email)

#     # Kiểm tra xem 'origin' có tồn tại không
#     if 'origin' in [remote.name for remote in repo.remotes]:
#         origin = repo.remotes.origin
#     else:
#         # Nếu 'origin' không tồn tại, thêm 'origin' với url github
#         origin = repo.create_remote('origin', github_url)

#     # Fetch changes từ remote repository
#     origin.fetch()
    
#     # Kiểm tra xem nhánh đã tồn tại hay chưa
#     if f"refs/heads/{branch_name}" in repo.refs:
#         print(f"Nhánh {branch_name} đã tồn tại trên repository.")
        
#         # Checkout branch đã tồn tại
#         repo.git.checkout(branch_name)
        
#         # Pull changes từ remote repository
#         repo.git.pull('--rebase', 'origin', branch_name)
        
#         return repo
    
#     # Nếu nhánh chưa tồn tại, tạo nhánh mới từ HEAD
#     branch = repo.create_head(branch_name, commit='HEAD')
    
#     # Checkout branch mới
#     branch.checkout()
    
#     # Set tracking branch
#     branch.set_tracking_branch(origin.refs[branch_name])
    
#     # Pull changes từ remote repository
#     repo.git.pull('--rebase', 'origin', branch_name)
    
#     # Thay đổi URL của origin để cung cấp thông tin đăng nhập
#     origin.set_url(github_url.replace('https://', f'https://{github_token}:x-oauth-basic@'))
    
#     # Push changes lên remote repository
#     repo.git.push('origin', branch_name)
    
#     print(f"Đã tồn tại repo: {repo_dir} trên Lakehouse!")

#     return repo


def pull_from_github(user_name, user_email, repo_dir, branch_name, github_token, github_url):
    '''
    Pull file from GitHub
    :param user_name: GitHub username
    :param user_email: GitHub email
    :param repo_dir: Path to the repo
    :param branch_name: Branch name
    :param github_token: GitHub token
    :param github_url: GitHub url
    :return: None

    Example:
    pull_from_github('nguyenvana', 'nguyenvana@gmail.com', 'C:\\Users\\nguyenvana\\Documents\\GitHub\\test', ['test.txt'], 'Test', 'main', 'ghp_1234567890', 'https://github.com/nguyenvana.vn/test.git')
    '''
    os.system(f'git config --global user.email {user_email}')
    os.system(f'git config --global user.name {user_name}')

    # Clone repo
    if not os.path.exists(repo_dir):
        clone_from_github(user_name, user_email, repo_dir, branch_name, github_token, github_url)
    else:
        repo = Repo(repo_dir)

        # Pull latest changes
        repo.git.pull(f'https://token:{github_token}@{github_url[8:]}', branch_name, allow_unrelated_histories=True, no_rebase=True)

    print(f"Đã kéo dữ liệu từ repo: {branch_name} trên GitHub về Lakehouse thành công!")


def push_to_github(user_name, user_email, repo_dir, file_name: list, commit_message, branch_name, github_token, github_url):
    '''
    Push file to GitHub
    :param user_name: GitHub username
    :param user_email: GitHub email
    :param repo_dir: Path to the repo
    :param file_name: List of file name
    :param commit_message: Commit message
    :param branch_name: Branch name
    :param github_token: GitHub token
    :param github_url: GitHub url
    :return: None

    Example:
    push_to_github('nguyenvana', 'nguyenvana@gmail.com', 'C:\\Users\\nguyenvana\\Documents\\GitHub\\test', ['test.txt'], 'Test', 'main', 'ghp_1234567890', 'https://github.com/nguyenvana.vn/test.git')
    '''
    pull_from_github(user_name, user_email, repo_dir, branch_name, github_token, github_url)
    
    repo = Repo(repo_dir)
    
    # Thêm file vào stage
    for file in file_name:
        repo.git.add(file)

    # Commit
    author = Actor(user_name, user_email)
    repo.index.commit(commit_message, author=author)

    # Pull latest changes
    # repo.git.pull(f'https://token:{github_token}@{github_url[8:]}', branch_name, allow_unrelated_histories=True, rebase=True)

    # Push commit
    repo.git.push(f'https://token:{github_token}@{github_url[8:]}', f'HEAD:refs/heads/{branch_name}')
    print(f"Đã đẩy dữ liệu từ repo trên Lakehouse sang Repo_github:{branch_name} thành công!")

def delete_to_github(repo_name, file_name, github_token):
    '''
    delete file to GitHub
    :param repo_name: repo_name
    :param github_token: GitHub token
    Example:
    delete_to_github(frabric_github, 'test.txt, 'ghp_1234567890')
    '''

    # Khởi tạo đối tượng Github
    github = Github(github_token)

    # Lấy thông tin repository
    repo = github.get_user().get_repo(repo_name)

    # Xoá file từ repository
    contents = repo.get_contents(file_name)
    repo.delete_file(contents.path, "Delete file", contents.sha)

    print(f"File: {file_name} đã được xoá thành công trên GitHub!")