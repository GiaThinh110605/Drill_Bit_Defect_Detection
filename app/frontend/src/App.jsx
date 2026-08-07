import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, X, AlertCircle, RefreshCw } from 'lucide-react';
import './App.css'; // Optional if you have other styles, otherwise index.css handles it

const API_URL = import.meta.env.VITE_API_URL || '';

function App() {
  const [mode, setMode] = useState('single'); // 'single' or 'compare'
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [renderSize, setRenderSize] = useState({ width: 0, height: 0 });

  // Comparison mode state
  const [compareDetections, setCompareDetections] = useState({ before: [], after: [] });

  const imageRef = useRef(null);
  const fileInputRef = useRef(null);

  // Update render size when image loads or window resizes
  useEffect(() => {
    const handleResize = () => {
      if (imageRef.current) {
        setRenderSize({
          width: imageRef.current.clientWidth,
          height: imageRef.current.clientHeight
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [preview]);

  const onImageLoad = (e) => {
    setImageSize({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight
    });
    setRenderSize({
      width: e.target.clientWidth,
      height: e.target.clientHeight
    });
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      setError('Vui lòng chọn một file hình ảnh.');
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setDetections([]);
    setError(null);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInputRef.current.files = e.dataTransfer.files;
      handleFileChange({ target: fileInputRef.current });
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file_name', file); // 'file_name' matching FastAPI parameter

    try {
      if (mode === 'compare') {
        const response = await axios.post('/api/compare', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setCompareDetections({
          before: response.data.before_aug,
          after: response.data.after_aug
        });
      } else {
        const response = await axios.post('/api/predict', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setDetections(response.data);
      }
    } catch (err) {
      console.error(err);
      setError('Đã xảy ra lỗi khi phân tích ảnh. Vui lòng kiểm tra server FastAPI.');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setDetections([]);
    setCompareDetections({ before: [], after: [] });
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Hàm tính toán màu cho từng loại lỗi
  const getColorForClass = (className) => {
    const colors = {
      'Broken': '#ef4444',     // Red
      'Chipped': '#f59e0b',    // Orange
      'Scratched': '#eab308',  // Yellow
      'Severe_Rust': '#84cc16',// Lime
      'Tip_Wear': '#3b82f6',   // Blue
    };
    return colors[className] || '#a855f7'; // Purple default
  };

  // Tính scale tỷ lệ để vẽ box chính xác lên ảnh (xử lý object-fit: contain)
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;

  if (imageSize.width > 0 && imageSize.height > 0 && renderSize.width > 0 && renderSize.height > 0) {
    // Tỷ lệ scale thực tế của ảnh khi dùng object-fit: contain
    scale = Math.min(
      renderSize.width / imageSize.width,
      renderSize.height / imageSize.height
    );

    // Kích thước thực tế của ảnh sau khi scale
    const renderedWidth = imageSize.width * scale;
    const renderedHeight = imageSize.height * scale;

    // Khoảng cách bị dư ra (viền đen) do object-fit: contain (nằm giữa)
    offsetX = (renderSize.width - renderedWidth) / 2;
    offsetY = (renderSize.height - renderedHeight) / 2;
  }

  return (
    <div className="app-container">
      <header>
        <h1>Drill Bit Defect Detector</h1>
        <p>Phân tích và phát hiện tự động các lỗi trên mũi khoan sử dụng YOLOv12</p>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
            onClick={() => setMode('single')}
          >
            Đơn ảnh
          </button>
          <button
            className={`mode-btn ${mode === 'compare' ? 'active' : ''}`}
            onClick={() => setMode('compare')}
          >
            So sánh mô hình
          </button>
        </div>
      </header>

      {!preview ? (
        <div className="glass-panel">
          <div
            className="upload-zone"
            onClick={() => fileInputRef.current.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="upload-icon" />
            <div className="upload-text">Kéo thả ảnh vào đây hoặc click để chọn</div>
            <div className="upload-hint">Hỗ trợ JPG, PNG, JPEG</div>
            <input
              type="file"
              className="hidden-input"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
            />
          </div>
          {error && (
            <div style={{ color: 'var(--danger)', marginTop: '1rem', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <AlertCircle size={18} /> {error}
            </div>
          )}
        </div>
      ) : (
        <div className="results-grid">
          {/* Cột hiển thị ảnh */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>

            {/* Vùng chứa có background đen */}
            <div style={{ width: '100%', backgroundColor: '#000', borderRadius: '8px', display: 'flex', justifyContent: 'center', padding: '10px' }}>

              {/* Wrapper ôm sát kích thước thực tế của ảnh */}
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <img
                  src={preview}
                  alt="Preview"
                  ref={imageRef}
                  onLoad={onImageLoad}
                  style={{ display: 'block', maxWidth: '100%', maxHeight: '600px' }}
                />

                {/* Overlay Bounding Boxes */}
                {!loading && renderSize.width > 0 && detections.map((det, idx) => {
                  const [x1, y1, x2, y2] = det.box;
                  const color = getColorForClass(det.class_name);

                  // Do wrapper ôm sát ảnh, ta chỉ cần chia tỷ lệ đơn giản
                  const scaleX = renderSize.width / imageSize.width;
                  const scaleY = renderSize.height / imageSize.height;

                  const left = x1 * scaleX;
                  const top = y1 * scaleY;
                  const width = (x2 - x1) * scaleX;
                  const height = (y2 - y1) * scaleY;

                  return (
                    <div
                      key={idx}
                      className="detection-box"
                      style={{
                        left: `${left}px`,
                        top: `${top}px`,
                        width: `${width}px`,
                        height: `${height}px`,
                        borderColor: color
                      }}
                    >
                      <div className="detection-label" style={{ backgroundColor: color }}>
                        {det.class_name} {(det.conf * 100).toFixed(1)}%
                      </div>
                    </div>
                  );
                })}

                {loading && (
                  <div className="loading-overlay">
                    <div className="spinner"></div>
                    <div>Đang phân tích mũi khoan...</div>
                  </div>
                )}
              </div>
            </div>

            <div className="actions" style={{ width: '100%' }}>
              <button
                className="btn btn-primary"
                onClick={analyzeImage}
                disabled={loading || (mode === 'single' ? detections.length > 0 : compareDetections.before.length > 0 || compareDetections.after.length > 0)}
                style={{ flex: 1 }}
              >
                {mode === 'compare' ? 'So sánh mô hình' : 'Phân tích ảnh'}
              </button>
              <button className="btn btn-secondary" onClick={resetAll} disabled={loading}>
                <RefreshCw size={18} />
                Thử ảnh khác
              </button>
            </div>

            {error && <div style={{ color: 'var(--danger)', textAlign: 'center' }}>{error}</div>}
          </div>

          {/* Cột thống kê */}
          <div className="stats-panel">
            {mode === 'single' ? (
              <>
                <div className="stat-card">
                  <div className="stat-label">Tổng số lỗi phát hiện</div>
                  <div className={`stat-value ${detections.length > 0 ? 'danger' : (detections.length === 0 && !loading && imageSize.width > 0 && !error && file ? 'success' : '')}`}>
                    {detections.length}
                  </div>
                </div>

                {detections.length > 0 && (
                  <div className="glass-panel" style={{ flex: 1 }}>
                    <h3 style={{ marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--panel-border)' }}>
                      Chi tiết lỗi
                    </h3>
                    <div className="detections-list">
                      {detections.map((det, idx) => {
                        const color = getColorForClass(det.class_name);
                        return (
                          <div key={idx} className="detection-item" style={{ borderLeftColor: color }}>
                            <span className="detection-name" style={{ color: color }}>{det.class_name}</span>
                            <span className="detection-conf">Độ tin cậy: {(det.conf * 100).toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {detections.length === 0 && !loading && imageSize.width > 0 && !error && file && (
                  <div className="glass-panel" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--success)' }}>
                    Mũi khoan hoàn hảo, không phát hiện lỗi!
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="stat-card">
                  <div className="stat-label">Trước khi điều chỉnh tham số</div>
                  <div className="stat-value">{compareDetections.before.length}</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Sau khi điều chỉnh tham số</div>
                  <div className="stat-value">{compareDetections.after.length}</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Chênh lệch</div>
                  <div className={`stat-value ${compareDetections.after.length - compareDetections.before.length >= 0 ? 'success' : 'danger'}`}>
                    {compareDetections.after.length - compareDetections.before.length}
                  </div>
                </div>

                {compareDetections.before.length > 0 && (
                  <div className="glass-panel" style={{ flex: 1 }}>
                    <h3 style={{ marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--panel-border)' }}>
                      Trước khi tăng cường
                    </h3>
                    <div className="detections-list">
                      {compareDetections.before.map((det, idx) => {
                        const color = getColorForClass(det.class_name);
                        return (
                          <div key={idx} className="detection-item" style={{ borderLeftColor: color }}>
                            <span className="detection-name" style={{ color: color }}>{det.class_name}</span>
                            <span className="detection-conf">Độ tin cậy: {(det.conf * 100).toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {compareDetections.after.length > 0 && (
                  <div className="glass-panel" style={{ flex: 1 }}>
                    <h3 style={{ marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--panel-border)' }}>
                      Sau khi tăng cường
                    </h3>
                    <div className="detections-list">
                      {compareDetections.after.map((det, idx) => {
                        const color = getColorForClass(det.class_name);
                        return (
                          <div key={idx} className="detection-item" style={{ borderLeftColor: color }}>
                            <span className="detection-name" style={{ color: color }}>{det.class_name}</span>
                            <span className="detection-conf">Độ tin cậy: {(det.conf * 100).toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Comparison mode: Show two side-by-side images */}
          {mode === 'compare' && preview && (
            <div className="glass-panel" style={{ gridColumn: '1 / -1', marginTop: '1rem' }}>
              <h3 style={{ marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--panel-border)' }}>
                So sánh kết quả
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                {/* Before augmentation */}
                <div>
                  <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Trước khi tăng cường</h4>
                  <div style={{ width: '100%', backgroundColor: '#000', borderRadius: '8px', display: 'flex', justifyContent: 'center', padding: '10px' }}>
                    <div style={{ position: 'relative', display: 'inline-block' }}>
                      <img
                        src={preview}
                        alt="Before Augmentation"
                        style={{ display: 'block', maxWidth: '100%', maxHeight: '400px' }}
                      />
                      {compareDetections.before.map((det, idx) => {
                        const [x1, y1, x2, y2] = det.box;
                        const color = getColorForClass(det.class_name);
                        const scaleX = 1; // Simplified for comparison
                        const scaleY = 1;

                        return (
                          <div
                            key={idx}
                            className="detection-box"
                            style={{
                              left: `${x1 * scaleX}px`,
                              top: `${y1 * scaleY}px`,
                              width: `${(x2 - x1) * scaleX}px`,
                              height: `${(y2 - y1) * scaleY}px`,
                              borderColor: color
                            }}
                          >
                            <div className="detection-label" style={{ backgroundColor: color }}>
                              {det.class_name} {(det.conf * 100).toFixed(1)}%
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* After augmentation */}
                <div>
                  <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Sau khi tăng cường</h4>
                  <div style={{ width: '100%', backgroundColor: '#000', borderRadius: '8px', display: 'flex', justifyContent: 'center', padding: '10px' }}>
                    <div style={{ position: 'relative', display: 'inline-block' }}>
                      <img
                        src={preview}
                        alt="After Augmentation"
                        style={{ display: 'block', maxWidth: '100%', maxHeight: '400px' }}
                      />
                      {compareDetections.after.map((det, idx) => {
                        const [x1, y1, x2, y2] = det.box;
                        const color = getColorForClass(det.class_name);
                        const scaleX = 1; // Simplified for comparison
                        const scaleY = 1;

                        return (
                          <div
                            key={idx}
                            className="detection-box"
                            style={{
                              left: `${x1 * scaleX}px`,
                              top: `${y1 * scaleY}px`,
                              width: `${(x2 - x1) * scaleX}px`,
                              height: `${(y2 - y1) * scaleY}px`,
                              borderColor: color
                            }}
                          >
                            <div className="detection-label" style={{ backgroundColor: color }}>
                              {det.class_name} {(det.conf * 100).toFixed(1)}%
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
