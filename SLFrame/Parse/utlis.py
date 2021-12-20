def setParseAttribute(parse, d):
    """
    parse: 解析器
    d: 配置信息字典
    将字典中的数据存入parse的成员属性中
    """
    # for item in vars(parse):
    #     if item in d:
    #         setattr(parse, item, d[item])
    for item in d:
        if item in vars(parse):
            setattr(parse, item, d[item])
        else:
            parse.kwargs[item] = d[item]
