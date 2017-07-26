# unrolled RNN network
rnn.unroll <- function(num.rnn.layer, seq.len, input.size,
                       num.embed, num.hidden, num.label,
                       dropout = 0,
                       ignore_label = 0,
                       init.state = NULL,
                       config,
                       cell.type = "lstm",
                       output.last.state = FALSE) {
  embed.weight <- mx.symbol.Variable("embed.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  param.cells <- lapply(1:num.rnn.layer, function(i) {
    if (cell.type == "lstm") {
      cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                   i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                   h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                   h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
    } else if (cell.type == "gru") {
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                   gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                   gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                   gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                   trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                   trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                   trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                   trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
    }
    return (cell)
  })
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  #data_mask <- mx.symbol.Variable("data_mask")
  data.mask.array <- mx.symbol.Variable("data.mask.array")
  data.mask.array <- mx.symbol.stop_gradient(data.mask.array, name = "data.mask.array")
  
  embed <- mx.symbol.Embedding(data = data, input_dim = input.size,
                               weight = embed.weight,
                               output_dim = num.embed,
                               name = "embed")
  
  wordvec <- mx.symbol.split(data = embed, axis = 1,
                             num.outputs = seq.len,
                             squeeze_axis = TRUE)
  data_mask_split <- mx.symbol.split(data = data.mask.array, axis = 1,
                                     num.outputs = seq.len,
                                     squeeze_axis = TRUE)
  
  last.hidden <- list()
  last.states <- list()
  decode <- list()
  softmax <- list()
  fc <- list()
  
  for (seqidx in 1:seq.len) {
    hidden <- wordvec[[seqidx]]
    
    for (i in 1:num.rnn.layer) {
      if (seqidx == 1) {
        prev.state <- init.state[[i]]
      } else{
        prev.state <- last.states[[i]]
      }
      
      if (cell.type == "lstm") {
        cell.symbol <- lstm.cell
      } else if (cell.type == "gru") {
        cell.symbol <- gru.cell
      }
      
      next.state <- cell.symbol(num.hidden = num.hidden,
                                indata = hidden,
                                prev.state = prev.state,
                                param = param.cells[[i]],
                                seqidx = seqidx,
                                layeridx = i,
                                dropout = dropout,
                                data_masking = data_mask_split[[seqidx]])
      hidden <- next.state$h
      #if (dropout > 0) hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
      last.states[[i]] <- next.state
    }
    
    # Decoding
    if (config == "one-to-one") {
      last.hidden <- c(last.hidden, hidden)
    }
  }
  
  if (config == "seq-to-one") {
    fc <- mx.symbol.FullyConnected(data = hidden,
                                   weight = cls.weight,
                                   bias = cls.bias,
                                   num.hidden = num.label)
    
    loss <- mx.symbol.SoftmaxOutput(data = fc,
                                    name = "sm",
                                    label = label,
                                    ignore_label = ignore_label)
    
  } else if (config == "one-to-one") {
    last.hidden_expand = lapply(last.hidden, function(i) mx.symbol.expand_dims(i, axis = 1))
    concat <- mx.symbol.concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape = mx.symbol.Reshape(concat, shape = c(num.hidden,-1))
    
    fc <- mx.symbol.FullyConnected(data = reshape,
                                   weight = cls.weight,
                                   bias = cls.bias,
                                   num.hidden = num.label)
    
    label <- mx.symbol.reshape(data = label, shape = c(-1))
    loss <- mx.symbol.SoftmaxOutput(data = fc,
                                    name = "sm",
                                    label = label,
                                    ignore_label = ignore_label)
  } else {
    stop("Unsupported config. Please use seq-to-one or one-to-one.")
  }
  
  if (output.last.state) {
    group <- mx.symbol.Group(c(unlist(last.states), loss))
    return(group)
  } else {
    return(loss)
  }
}

mx.rnn.buckets.train <- function(train.data,
                                 eval.data = NULL,
                                 num.rnn.layer,
                                 num.hidden,
                                 num.embed,
                                 num.label,
                                 input.size,
                                 ctx = NULL,
                                 begin.round = 1,
                                 num.round = 1,
                                 initializer = mx.init.uniform(0.01),
                                 dropout = 0,
                                 config = "one-to-one",
                                 kvstore = "local",
                                 optimizer = 'sgd',
                                 batch.end.callback = NULL,
                                 epoch.end.callback = NULL,
                                 metric = mx.metric.rmse,
                                 cell.type = "lstm",
                                 input.names = c("data", "data.mask.array"),
                                 output.names = NULL,
                                 arg.params = NULL,
                                 aux.params = NULL,
                                 fixed.param = NULL,
                                 verbose = TRUE) {
  
  if (class(train.data) != "BucketIter") {
    stop("BucketIter is required.")
  }
  if (!train.data$iter.next()) {
    train.data$reset()
    if (!train.data$iter.next()) stop("Empty train.data")
  }
  if (!is.null(eval.data)) {
    if (class(eval.data) != "BucketIter") {
      stop("BucketIter is required.")
    }
    if (!eval.data$iter.next()) {
      eval.data$reset()
      if (!eval.data$iter.next()) stop("Empty eval.data")
    }
  }

  # get unrolled lstm symbol
  if (verbose) message(paste0("Seq len: ", paste(train.data$bucket.names, collapse = " ")))
  sym_list <- sapply(train.data$bucket.names, function(x) {
    rnn.unroll(num.rnn.layer = num.rnn.layer,
               num.hidden = num.hidden,
               seq.len = as.integer(x),
               input.size = input.size,
               num.embed = num.embed,
               num.label = num.label,
               dropout = dropout,
               cell.type = cell.type,
               config = config)
  }, simplify = FALSE, USE.NAMES = TRUE)
  
  symbol <- sym_list[[names(train.data$bucketID)]]
  
  if (is.null(input.names)) {
    input.names <- "data"
  }
  input.shape <- sapply(input.names, function(n){dim(train.data$value()[[n]])}, simplify = FALSE)
  if (is.null(output.names)) {
    arg_names <- arguments(symbol)
    output.names <- arg_names[endsWith(arg_names, "label")]
    output.shape <- list()
    output.shape[[output.names]] <- dim((train.data$value())$label)
  } else {
    output.shape <- sapply(output.names, function(n){dim(train.data$value()[[n]])}, simplify = FALSE)  
  }
  
  params <- mx.model.init.params(symbol = symbol, input.shape, output.shape, initializer, mx.cpu())
  if (!is.null(arg.params)) params$arg.params <- arg.params
  if (!is.null(aux.params)) params$aux.params <- aux.params
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) ctx <- list(ctx)
  if (!is.list(ctx)) stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    if (is.numeric(input.shape)) {
      ndim <- length(input.shape)
      batchsize = input.shape[[ndim]]      
    } else {
      ndim <- length(input.shape[[1]])
      batchsize = input.shape[[1]][[ndim]]
    }
    optimizer <- mx.opt.create(optimizer, rescale.grad=(1/batchsize), ...)
  }
  kvstore <- mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), verbose = verbose)
  
  model <- mx.model.train(symbol = sym_list,
                          ctx = ctx,
                          input.shape = input.shape,
                          output.shape = output.shape,
                          arg.params = params$arg.params,
                          aux.params = params$aux.params,
                          begin.round = begin.round,
                          end.round = num.round,
                          optimizer = optimizer,
                          train.data = train.data,
                          eval.data = eval.data,
                          metric = metric,
                          epoch.end.callback = epoch.end.callback,
                          batch.end.callback = batch.end.callback,
                          kvstore = kvstore,
                          fixed.param = fixed.param,
                          verbose = verbose,
                          is.bucket = TRUE)
  return(model)
}

mx.rnn.buckets.infer <- function(data.iter,
                                 model,
                                 config,
                                 ctx = NULL,
                                 kvstore = "local",
                                 output.last.state = FALSE,
                                 init.state = NULL,
                                 cell.type = "lstm",
                                 input.names = c("data", "data.mask.array"),
                                 output.names = NULL) {
  if (class(data.iter) != "BucketIter") {
    stop("BucketIter is required.")
  }
  
  if (!data.iter$iter.next()) {
    data.iter$reset()
    if (!data.iter$iter.next()) stop("Empty data")
  }
  
  if (cell.type == "lstm") {
    num.rnn.layer = round((length(model$arg.params) - 3) / 4)
    num.hidden = dim(model$arg.params$l1.h2h.weight)[1]
  } else if (cell.type == "gru") {
    num.rnn.layer = round((length(model$arg.params) - 3) / 8)
    num.hidden = dim(model$arg.params$l1.gates.h2h.weight)[1]
  }
  
  input.size = dim(model$arg.params$embed.weight)[2]
  num.embed = dim(model$arg.params$embed.weight)[1]
  num.label = dim(model$arg.params$cls.bias)
  
  sym_list <- sapply(data.iter$bucket.names, function(x) {
    mxnet:::rnn.unroll(num.rnn.layer = num.rnn.layer,
               num.hidden = num.hidden,
               seq.len = as.integer(x),
               input.size = input.size,
               num.embed = num.embed,
               num.label = num.label,
               config = config,
               init.state = init.state,
               cell.type = cell.type,
               output.last.state = output.last.state)
  }, simplify = FALSE, USE.NAMES = TRUE)
  
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) ctx <- list(ctx)
  if (!is.list(ctx)) stop("ctx must be mx.context or list of mx.context")

  if (is.null(input.names)) {
    input.names <- "data"
  }
  
  if (is.null(output.names)) {
    symbol <- sym_list[[names(data.iter$bucketID)]]
    arg_names <- arguments(symbol)
    output.names <- arg_names[endsWith(arg_names, "label")]
  }
  
  input.shape <- sapply(input.names, function(n){dim(data.iter$value()[[n]])}, simplify = FALSE)
  output.shape <- sapply(output.names, function(n){dim(data.iter$value()[[n]])}, simplify = FALSE)  
  
  input.update <- sapply(input.names, function(n) {
    mx.nd.zeros(input.shape[[n]], ctx[[1]])
  }, simplify = FALSE, USE.NAMES = TRUE)
  
  output.update <- sapply(output.names, function(n) {
    mx.nd.zeros(output.shape[[n]], ctx[[1]])
  }, simplify = FALSE, USE.NAMES = TRUE)
  
  model$arg.params[names(input.update)] <- input.update
  model$arg.params[names(output.update)] <- output.update
  
  #train.execs <- lapply(1:ndevice, function(i) {
  arg_lst <- list(symbol = symbol, ctx = ctx[[1]], grad.req = "write")
  arg_lst <- append(arg_lst, input.shape)
  arg_lst <- append(arg_lst, output.shape)
  arg_lst[["fixed.param"]] = fixed.param
  pexec <- do.call(mx.simple.bind, arg_lst)
  #})
  # set the parameters into executors
  #for (texec in train.execs) {
  mx.exec.update.arg.arrays(pexec, model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(pexec, model$aux.params, match.name = TRUE)
  #}
  
  data.iter$reset()
  packer <- mxnet:::mx.nd.arraypacker()
  while (data.iter$iter.next()) {
    dlist <- data.iter$value()
    dlist <- dlist[names(dlist) %in% arguments(symbol)]
    dlist <- dlist[names(dlist) %in% input.names]
    #slices <- lapply(1:ndevice, function(i) {
    #  s <- input_slice[[i]]
    #  ret <- sapply(names(dlist), function(n) {mxnet:::mx.nd.slice(dlist[[n]], s$begin, s$end)})
    #  return(ret)
    #})
    
    symbol <- sym_list[[names(data.iter$bucketID)]]
    
    input.shape <- sapply(input.names, function(n) {dim(data.iter$value()[[n]])}, simplify = FALSE)
    output.shape[[output.names]] <- dim((data.iter$value())$label)
    #input_slice <- mxnet:::mx.model.slice.shape(input.shape, ndevice)
    #output_slice <- mxnet:::mx.model.slice.shape(output.shape, ndevice)
    
    #train.execs <- lapply(1:ndevice, function(i) {
    arg_lst <- list(symbol = symbol, ctx = ctx[[1]], grad.req = "write")
    arg_lst <- append(arg_lst, input.shape)
    arg_lst <- append(arg_lst, output.shape)
    arg_lst[["fixed.param"]] <- fixed.param
    
    input.update <- sapply(input.names, function(n) {mx.nd.zeros(input.shape[[n]], ctx[[1]])}, simplify = FALSE, USE.NAMES = TRUE)
    
    tmp <- pexec$arg.arrays
    tmp[names(input.update)] <- input.update
    arg_lst[["arg.arrays"]] <- tmp
    arg_lst[["aux.arrays"]] <- pexec$aux.arrays
    pexec <- do.call(mx.simple.bind, arg_lst)
    #})
    
    mx.exec.update.arg.arrays(pexec, dlist, match.name = TRUE)
    
    #for (i in 1:ndevice) {
    #  s <- slices[[i]]
    #  names(s)[endsWith(names(s), "label")] = arguments(symbol)[endsWith(arguments(symbol), "label")]
    #  s <- s[names(s) %in% arguments(symbol)]
    #  mx.exec.update.arg.arrays(train.execs[[i]], s, match.name = TRUE)
    #}
    
    #for (texec in train.execs) {
    mx.exec.forward(pexec, is.train = FALSE)
    #}
    
    out.preds <- mx.nd.copyto(pexec$ref.outputs[[1]], mx.cpu())
    
    #for (i in 1 : ndevice) {
    packer$push(out.preds)
    #}
  }
  data.iter$reset()
  return(packer$get())
}