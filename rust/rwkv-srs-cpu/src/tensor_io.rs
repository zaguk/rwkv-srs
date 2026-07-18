use candle_core::{bail, Device, Result, Tensor};

pub(crate) type Tensor2List = Vec<Vec<f32>>;
pub(crate) type Tensor3List = Vec<Vec<Vec<f32>>>;
pub(crate) type Tensor4List = Vec<Vec<Vec<Vec<f32>>>>;
pub(crate) type Tensor5List = Vec<Vec<Vec<Vec<Vec<f32>>>>>;

pub(crate) fn tensor_from_1d(values: Vec<f32>, expected_len: usize, name: &str) -> Result<Tensor> {
    if values.len() != expected_len {
        bail!(
            "{name} expected length {expected_len}, got {}",
            values.len()
        );
    }
    Tensor::from_vec(values, (expected_len,), &Device::Cpu)
}

pub(crate) fn tensor_from_2d(values: Tensor2List, name: &str) -> Result<Tensor> {
    let rows = values.len();
    if rows == 0 {
        bail!("{name} must not be empty");
    }
    let cols = values[0].len();
    if cols == 0 {
        bail!("{name} rows must not be empty");
    }
    if values.iter().any(|row| row.len() != cols) {
        bail!("{name} must be rectangular");
    }
    let flat = values.into_iter().flatten().collect::<Vec<_>>();
    Tensor::from_vec(flat, (rows, cols), &Device::Cpu)
}

pub(crate) fn tensor_from_3d(values: Tensor3List, name: &str) -> Result<Tensor> {
    let dim0 = values.len();
    if dim0 == 0 {
        bail!("{name} must not be empty");
    }
    let dim1 = values[0].len();
    if dim1 == 0 {
        bail!("{name} second dimension must not be empty");
    }
    let dim2 = values[0][0].len();
    if dim2 == 0 {
        bail!("{name} third dimension must not be empty");
    }
    for rows in &values {
        if rows.len() != dim1 {
            bail!("{name} must be rectangular");
        }
        for row in rows {
            if row.len() != dim2 {
                bail!("{name} must be rectangular");
            }
        }
    }
    let flat = values.into_iter().flatten().flatten().collect::<Vec<_>>();
    Tensor::from_vec(flat, (dim0, dim1, dim2), &Device::Cpu)
}

pub(crate) fn tensor_from_4d(values: Tensor4List, name: &str) -> Result<Tensor> {
    let dim0 = values.len();
    if dim0 == 0 {
        bail!("{name} must not be empty");
    }
    let dim1 = values[0].len();
    if dim1 == 0 {
        bail!("{name} second dimension must not be empty");
    }
    let dim2 = values[0][0].len();
    if dim2 == 0 {
        bail!("{name} third dimension must not be empty");
    }
    let dim3 = values[0][0][0].len();
    if dim3 == 0 {
        bail!("{name} fourth dimension must not be empty");
    }
    for blocks in &values {
        if blocks.len() != dim1 {
            bail!("{name} must be rectangular");
        }
        for rows in blocks {
            if rows.len() != dim2 {
                bail!("{name} must be rectangular");
            }
            for row in rows {
                if row.len() != dim3 {
                    bail!("{name} must be rectangular");
                }
            }
        }
    }
    let flat = values.into_iter().flatten().flatten().flatten().collect();
    Tensor::from_vec(flat, (dim0, dim1, dim2, dim3), &Device::Cpu)
}

pub(crate) fn tensor_from_5d(values: Tensor5List, name: &str) -> Result<Tensor> {
    let dim0 = values.len();
    if dim0 == 0 {
        bail!("{name} must not be empty");
    }
    let dim1 = values[0].len();
    if dim1 == 0 {
        bail!("{name} second dimension must not be empty");
    }
    let dim2 = values[0][0].len();
    if dim2 == 0 {
        bail!("{name} third dimension must not be empty");
    }
    let dim3 = values[0][0][0].len();
    if dim3 == 0 {
        bail!("{name} fourth dimension must not be empty");
    }
    let dim4 = values[0][0][0][0].len();
    if dim4 == 0 {
        bail!("{name} fifth dimension must not be empty");
    }
    for blocks in &values {
        if blocks.len() != dim1 {
            bail!("{name} must be rectangular");
        }
        for heads in blocks {
            if heads.len() != dim2 {
                bail!("{name} must be rectangular");
            }
            for rows in heads {
                if rows.len() != dim3 {
                    bail!("{name} must be rectangular");
                }
                for row in rows {
                    if row.len() != dim4 {
                        bail!("{name} must be rectangular");
                    }
                }
            }
        }
    }
    let flat = values
        .into_iter()
        .flatten()
        .flatten()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    Tensor::from_vec(flat, (dim0, dim1, dim2, dim3, dim4), &Device::Cpu)
}

pub(crate) fn tensor_to_vec4(tensor: &Tensor) -> Result<Tensor4List> {
    let (dim0, dim1, dim2, dim3) = tensor.dims4()?;
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(dim0);
    let mut offset = 0;
    for _ in 0..dim0 {
        let mut blocks = Vec::with_capacity(dim1);
        for _ in 0..dim1 {
            let mut rows = Vec::with_capacity(dim2);
            for _ in 0..dim2 {
                rows.push(flat[offset..offset + dim3].to_vec());
                offset += dim3;
            }
            blocks.push(rows);
        }
        out.push(blocks);
    }
    Ok(out)
}

pub(crate) fn tensor_to_vec5(tensor: &Tensor) -> Result<Tensor5List> {
    let (dim0, dim1, dim2, dim3, dim4) = tensor.dims5()?;
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(dim0);
    let mut offset = 0;
    for _ in 0..dim0 {
        let mut block = Vec::with_capacity(dim1);
        for _ in 0..dim1 {
            let mut heads = Vec::with_capacity(dim2);
            for _ in 0..dim2 {
                let mut rows = Vec::with_capacity(dim3);
                for _ in 0..dim3 {
                    rows.push(flat[offset..offset + dim4].to_vec());
                    offset += dim4;
                }
                heads.push(rows);
            }
            block.push(heads);
        }
        out.push(block);
    }
    Ok(out)
}
