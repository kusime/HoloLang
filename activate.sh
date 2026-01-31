#!/bin/bash
# HoloLang 智能启动脚本
# 自动环境检测、创建、验证和服务启动

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目配置
ENV_NAME="hololang"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 日志函数
log_info() { echo -e "${BLUE}ℹ ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# 检测 conda 是否安装
check_conda() {
    log_info "检查 conda 是否安装..."
    if ! command -v conda &> /dev/null; then
        log_error "未找到 conda！请先安装 Miniconda 或 Anaconda"
        echo "   安装指南: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log_success "Conda 已安装: $(conda --version)"
}

# 检查环境是否存在
check_env_exists() {
    conda env list | grep -q "^${ENV_NAME} " && return 0 || return 1
}

# 创建 conda 环境
create_env() {
    log_info "创建 conda 环境: ${ENV_NAME}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    log_success "环境创建成功"
}

# 激活环境
activate_env() {
    log_info "激活环境: ${ENV_NAME}..."
    
    # 初始化 conda（确保 conda activate 可用）
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    
    log_success "环境已激活"
}

# 安装依赖
install_dependencies() {
    log_info "安装项目依赖..."
    
    if [ ! -f "${PROJECT_DIR}/requirements.txt" ]; then
        log_error "未找到 requirements.txt"
        exit 1
    fi
    
    pip install -r "${PROJECT_DIR}/requirements.txt" -q
    log_success "依赖安装完成"
}

# 验证关键依赖
verify_dependencies() {
    log_info "验证关键依赖..."
    
    local required_packages=("fastapi" "uvicorn" "pydantic" "minio" "whisperx")
    local missing_packages=()
    
    for pkg in "${required_packages[@]}"; do
        if ! python -c "import ${pkg}" 2>/dev/null; then
            missing_packages+=("${pkg}")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_warning "缺少依赖: ${missing_packages[*]}"
        return 1
    fi
    
    log_success "所有关键依赖已安装"
    return 0
}

# 检查环境变量
check_env_file() {
    log_info "检查环境配置文件..."
    
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        log_warning ".env 文件不存在"
        if [ -f "${PROJECT_DIR}/.env.example" ]; then
            log_info "从 .env.example 创建 .env..."
            cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
            log_success ".env 文件已创建，请根据需要修改配置"
        else
            log_warning "请手动创建 .env 文件配置环境变量"
        fi
    else
        log_success ".env 文件已存在"
    fi
}

# 加载环境变量
load_env_vars() {
    if [ -f "${PROJECT_DIR}/.env" ]; then
        export $(grep -v '^#' "${PROJECT_DIR}/.env" | xargs)
    fi
}

# 检查 TTS 服务
check_tts_service() {
    log_info "检查 TTS 服务连接..."
    
    local tts_url="${TTS_BASE_URL:-http://localhost:9880}"
    
    if curl -s --connect-timeout 3 "${tts_url}/health" > /dev/null 2>&1 || \
       curl -s --connect-timeout 3 "${tts_url}/" > /dev/null 2>&1; then
        log_success "TTS 服务可达: ${tts_url}"
        return 0
    else
        log_warning "TTS 服务不可达: ${tts_url}"
        log_warning "请确保 TTS 服务已启动"
        return 1
    fi
}

# 检查 MinIO 服务
check_minio_service() {
    log_info "检查 MinIO 服务连接..."
    
    local s3_endpoint="${S3_ENDPOINT:-localhost:9000}"
    local s3_secure="${S3_SECURE:-false}"
    
    if [ "${s3_secure}" = "true" ]; then
        local protocol="https"
    else
        local protocol="http"
    fi
    
    local minio_url="${protocol}://${s3_endpoint}/minio/health/live"
    
    if curl -s --connect-timeout 3 "${minio_url}" > /dev/null 2>&1; then
        log_success "MinIO 服务可达: ${s3_endpoint}"
        return 0
    else
        log_warning "MinIO 服务不可达: ${s3_endpoint}"
        log_warning "请确保 MinIO 服务已启动（可运行 docker-compose up -d）"
        return 1
    fi
}

# 运行健康检查
health_check() {
    log_info "运行健康检查..."
    
    local all_passed=true
    
    # 检查 Python 版本
    python_version=$(python --version 2>&1 | awk '{print $2}')
    log_success "Python 版本: ${python_version}"
    
    # 检查能否导入核心模块
    if python -c "from app.api.routes import app" 2>/dev/null; then
        log_success "核心模块导入正常"
    else
        log_error "核心模块导入失败，请检查代码"
        all_passed=false
    fi
    
    # 加载环境变量
    load_env_vars
    
    # 检查外部服务
    if ! check_tts_service; then
        all_passed=false
    fi
    
    if ! check_minio_service; then
        all_passed=false
    fi
    
    if [ "$all_passed" = false ]; then
        echo ""
        log_warning "部分健康检查未通过"
        echo ""
        read -p "$(echo -e ${YELLOW}▶${NC} 是否继续启动？外部服务不可用时 API 将返回错误 [y/N]: )" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "已取消启动"
            exit 0
        fi
    fi
    
    return 0
}

# 检查端口占用
check_port() {
    log_info "检查端口占用..."
    
    local port="${API_PORT:-8000}"
    
    if lsof -i:${port} -t > /dev/null 2>&1; then
        log_warning "端口 ${port} 已被占用"
        
        local pid=$(lsof -i:${port} -t)
        local process=$(ps -p ${pid} -o comm= 2>/dev/null || echo "未知进程")
        
        echo ""
        echo "   PID: ${pid}"
        echo "   进程: ${process}"
        echo ""
        
        read -p "$(echo -e ${YELLOW}▶${NC} 是否自动清理该进程？ [Y/n]: )" -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            log_info "正在停止进程 ${pid}..."
            kill ${pid} 2>/dev/null || kill -9 ${pid} 2>/dev/null
            sleep 1
            
            if lsof -i:${port} -t > /dev/null 2>&1; then
                log_error "进程清理失败，请手动运行: kill -9 ${pid}"
                return 1
            else
                log_success "端口 ${port} 已释放"
            fi
        else
            log_info "已取消启动，请手动释放端口 ${port}"
            exit 0
        fi
    else
        log_success "端口 ${port} 可用"
    fi
    
    return 0
}

# 启动服务器
start_server() {
    log_info "启动服务器..."
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  HoloLang TTS Pipeline Server${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo ""
    echo "📖 文档: http://localhost:8000/docs"
    echo "🔧 健康检查: http://localhost:8000/health"
    echo ""
    echo "按 Ctrl+C 停止服务器"
    echo ""
    
    cd "${PROJECT_DIR}"
    python -m app.api.routes
}

# 主流程
main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  HoloLang 智能启动脚本                ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""
    
    # 1. 检查 conda
    check_conda
    
    # 2. 检查/创建环境
    if check_env_exists; then
        log_success "Conda 环境 '${ENV_NAME}' 已存在"
    else
        log_warning "Conda 环境 '${ENV_NAME}' 不存在"
        create_env
    fi
    
    # 3. 激活环境
    activate_env
    
    # 4. 验证依赖
    if ! verify_dependencies; then
        log_warning "依赖不完整，正在安装..."
        install_dependencies
    fi
    
    # 5. 检查环境配置
    check_env_file
    
    # 6. 健康检查
    health_check
    
    echo ""
    log_success "环境准备完成！"
    echo ""
    
    # 7. 询问是否启动服务器
    read -p "$(echo -e ${YELLOW}▶${NC} 是否启动服务器？ [Y/n]: )" -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        # 检查端口占用
        check_port
        
        # 启动服务器
        start_server
    else
        echo ""
        log_info "环境已准备好，您可以手动运行："
        echo "  conda activate ${ENV_NAME}"
        echo "  python -m app.api.routes"
        echo ""
    fi
}

# 执行主流程
main "$@"
