function out = resampleImage(im,ref_,varargin)
% out = resampleImage(im,ref)
% out = resampleImage(im,ref,'interpolation',interp)
% out = resampleImage(im,0,'spacing',spacing)
%
% Fast resample an image using a reference image
% Works for ImageType and VectorImageType
%
%   interp is 'NN' (default), 'linear'

interp='NN';
userSpecifiedSpacing=false;
userSpecifiedSize=false;
userSpecifiedCentre=false;
spacing = [-1 -1 -1];
doblurring=false;
enlargeBoders=false;
MAX_CHUNK_SIZE = 300;
identityOrientation = false;
for i=1:size(varargin,2)
    if (strcmp(varargin{i},'interpolation'))
        interp=varargin{i+1};
    elseif (strcmp(varargin{i},'spacing'))
        spacing=varargin{i+1};
        userSpecifiedSpacing=true;
    elseif (strcmp(varargin{i},'spacing_and_size'))
        spacing_and_size=varargin{i+1};
        spacing=spacing_and_size(1:end/2);
        sz=spacing_and_size(end/2+1:end);
        userSpecifiedSpacing=true;
        userSpecifiedSize = true;
    elseif (strcmp(varargin{i},'spacing_and_size_and_centre'))
        spacing_and_size=varargin{i+1};
        spacing=spacing_and_size(1:end/3);
        sz=spacing_and_size(end/3+1:2*end/3);
        centre=spacing_and_size(2*end/3+1:end);
        userSpecifiedSpacing=true;
        userSpecifiedSize = true;
        userSpecifiedCentre = true;
    elseif (strcmp(varargin{i},'identityOrientation'))
        identityOrientation=true;
    elseif (strcmp(varargin{i},'blur'))
        doblurring = true;
        blur_neighourhood=varargin{i+1};
        blur_sigma=varargin{i+2};
    elseif (strcmp(varargin{i},'maxChunkSize'))
        MAX_CHUNK_SIZE = varargin{i+1};
        i=i+1;
    elseif (strcmp(varargin{i},'enlargeBorders'))
        enlargeBoders=true;
        i=i+1;
    end
    
end
%----------------------------

if identityOrientation
    bounds = im.GetBounds();
        
    if ~userSpecifiedSpacing
        spacing = im.spacing;
    end
    newSize = floor((bounds([2:2:end])-bounds([1:2:end]))./spacing(:))+1;
    ref_ = ImageType(newSize,bounds([1:2:end]),spacing,eye(numel(bounds)/2));
    userSpecifiedSpacing = false;
end

if userSpecifiedSpacing
    newSize = floor(im.size(:).*im.spacing(:)./spacing(:));
    origin = im.origin;
    if userSpecifiedSize
       % now make sure the size is the specified size without changing the spacing.
       padd_remove = newSize - sz;
       origin = im.origin +ceil(padd_remove/2).*spacing;
       newSize = sz;
    end
    if userSpecifiedCentre
       origin = centre-(newSize-1)/2.*spacing; 
    end    
    % positions matrix
    ref = ImageType(newSize,origin, spacing,im.orientation);
else
    ref = PatchType(ref_);
end



if (isa(im,'VectorImageType') || isfield(im,'datax'))
    im_ = VectorImageType(im) ;
    im_.datax  = im.datax ;
    im_.datay  = im.datay ;
    im_.dataz  = im.dataz ;
else
    im_ = PatchType(im) ;
    im_.data  = im.data ;
end
ndims_ = ref.ndimensions;
if doblurring
    
    for i=1:ndims_
        % This will ensure even number of elements in the kernel
        N = blur_neighourhood(i);
        k= gausswin(N,2/blur_sigma(i));
        
        nelems = ones(1,ndims_);
        nelems(i)=N;
        
        kernel = ones(nelems);
        kernel(:)=k/sum(k(:));
        
        im_.data = convn(im_.data,kernel,'same');
        
        
    end
    
end

if enlargeBoders
    boundsIn = im_.GetBounds();
    boundsRef = ref.GetBounds();
    newBounds = boundsIn;
    newBounds(1:2:end) = min([boundsRef(1:2:end)  boundsIn(1:2:end)],[],2);
    newBounds(2:2:end) = max([boundsRef(2:2:end)  boundsIn(2:2:end)],[],2);
    newBounds=newBounds(:);
    
    
    
    ref = cropImage(ref,newBounds);
    
end

% -----------------------------Interpolate ----------------
% create the grid of the reference image


NCHUNKS = ceil(ref.size/MAX_CHUNK_SIZE);

if (ndims_==4)
    
    if (isa(im_,'VectorImageType') || isfield(im_,'datax'))
        out = VectorImageType(ref);
    else
        out = ImageType(ref);
    end
    
    chunked_size = ceil(out.size./NCHUNKS)';
    
    [ix, iy, iz, it]= ndgrid(0:NCHUNKS(1)-1,0:NCHUNKS(2)-1,0:NCHUNKS(3)-1,0:NCHUNKS(4)-1);
    intervals = [ix(:) iy(:) iz(:) it(:)];
    clear ix iy iz it;
    for i=1:size(intervals,1)
        ranges([1 3 5 7]) = intervals(i,:).*chunked_size+1;
        ranges([2 4 6 8]) = min([(intervals(i,:)+[1 1 1 1]).*chunked_size ; out.size']);
        
        ranges_size = ranges([2 4 6 8])-ranges([1 3 5 7])+[1 1 1 1];
        % generate all the indexes of the target image
        [x, y, z, t] = ndgrid( ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6),ranges(7):ranges(8));
        
        positions = ref.GetPosition([x(:) y(:) z(:) t(:)]');
        clear x y z t;
        datas = im_.GetValue(positions,interp);
        
        if isa(im_,'VectorImageType') || any(strcmp(properties(im_), 'datax'))
            out.datax(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6),ranges(7):ranges(8)) = reshape(datas(1,:),ranges_size);
            out.datay(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6),ranges(7):ranges(8)) = reshape(datas(2,:),ranges_size);
            out.dataz(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6),ranges(7):ranges(8)) = reshape(datas(3,:),ranges_size);
            
            data_ = im_.GetValue(positions,interp,'data');
            out.data(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6),ranges(7):ranges(8)) = ...
                reshape(data_,ranges_size);
            clear data positions;
        else
            out.data(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6),ranges(7):ranges(8)) = ...
                reshape(datas,ranges_size);
        end
        clear datas;
        
        
    end
    
elseif (ndims_==3)
    %[X Y Z]=ndgrid(1:ref.size(1),1:ref.size(2),1:ref.size(3));
    % Retrieve the world coordinates on the reference image
    
    if (isa(im_,'VectorImageType') || isfield(im_,'datax'))
        out = VectorImageType(ref);
    elseif isa(im_,'PatchType')
        out = PatchType(ref);
    else
        out = ImageType(ref);
    end
    
    chunked_size = ceil(out.size./NCHUNKS)';
    
    [ix, iy, iz]= ndgrid(0:NCHUNKS(1)-1,0:NCHUNKS(2)-1,0:NCHUNKS(3)-1);
    intervals = [ix(:) iy(:) iz(:)];
    clear ix iy iz;
    for i=1:size(intervals,1)
        ranges([1 3 5]) = intervals(i,:).*chunked_size+1;
        ranges([2 4 6]) = min([(intervals(i,:)+[1 1 1]).*chunked_size ; out.size']);
        
        ranges_size = ranges([2 4 6])-ranges([1 3 5])+[1 1 1];
        % generate all the indexes of the target image
        [x, y, z] = ndgrid( ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6));
        
        positions = ref.GetPosition([x(:) y(:) z(:)]');
        clear x y z;
        datas = im_.GetValue(positions,interp);
        
        if isa(im_,'VectorImageType') || any(strcmp(properties(im_), 'datax'))
            out.datax(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6)) = reshape(datas(1,:),ranges_size);
            out.datay(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6)) = reshape(datas(2,:),ranges_size);
            out.dataz(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6)) = reshape(datas(3,:),ranges_size);
            
            data_ = im_.GetValue(positions,interp,'data');
            out.data(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6)) = ...
                reshape(data_,ranges_size);
            clear data positions;
        else
            out.data(ranges(1):ranges(2),ranges(3):ranges(4),ranges(5):ranges(6)) = ...
                reshape(datas,ranges_size);
        end
        clear datas;
        
        
    end
elseif(ndims_==2)
    [X, Y]=ndgrid(1:ref.size(1),1:ref.size(2));
    % Retrieve the world coordinates on the reference image
    positions = ref.GetPosition([X(:) Y(:)]');
    
    if isa(im_,'VectorImageType') ||  any(strcmp(properties(im_), 'datax'))
        out = VectorImageType(ref);
        datas = im_.GetValue(positions,interp);
        out.datax = reshape(datas(1,:),ref.size(1),ref.size(2));
        out.datay = reshape(datas(2,:),ref.size(1),ref.size(2));
        % out.dataz = reshape(datas(3,:),ref.size(1),ref.size(2));
    else
        out = ImageType(ref);
        out.data = reshape(im_.GetValue(positions,interp),ref.size(1),ref.size(2));
    end
    
end
out.data(out.data~=out.data)=out.paddingValue(1);

end
